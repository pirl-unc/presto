"""Canonical HAP1 consumer and fixed full-support trimming response model."""

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from presto.models.n_terminal_trimming import trimmed_log_amount

AA = "ACDEFGHIKLMNPQRSTVWY"
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Dataset:
    def __init__(self):
        self.cfg = read(ROOT / "reproduce/launch.json")
        recipe, baseline = Path(self.cfg["recipe_directory"]), Path(self.cfg["baseline_directory"])
        assert sha(recipe / "manifest.json") == self.cfg["recipe_manifest_sha256"]
        source_receipt = read(recipe / "manifest.json")
        for relative in [
            "derived/parents.parquet",
            "derived/precursor_inputs.parquet",
            "source/references.parquet",
            "source/occurrences.parquet",
        ]:
            assert sha(recipe / relative) == source_receipt["files"][relative]["sha256"]
        baseline_receipt = read(baseline / "results/closure.json")
        for relative, expected in baseline_receipt["artifact_sha256"].items():
            if relative.startswith(("results/fits/", "results/independent-")):
                assert sha(baseline / relative) == expected
        admission = read(baseline / "results/independent-admission.json")
        assert (
            admission["status"] == "passed"
            and admission["recipe_manifest_sha256"] == self.cfg["recipe_manifest_sha256"]
        )
        assert read(baseline / "results/independent-fit-verification.json")["status"] == "passed"
        # This cohort file contains only outer-training observations.
        self.rows = pq.read_table(
            baseline / "results/fits/cohort.parquet",
            filters=[("condition_id", "=", "hap1_erap1_ko")],
        ).to_pylist()
        assert len(self.rows) == 8281 and all(r["split"] == "train" for r in self.rows)
        self.peptides = sorted({r["peptide_sequence"] for r in self.rows})
        qi = {p: i for i, p in enumerate(self.peptides)}
        self.owner_rows = np.array([qi[r["peptide_sequence"]] for r in self.rows])
        self.target = np.array([r["target_log10_ratio"] for r in self.rows])
        self.baseline_input = np.array([r["precursor_log10_ratio"] for r in self.rows])
        self.fold = np.array(
            [
                int(
                    hashlib.sha256(
                        ("hap1_baseline_inner_v1:" + r["component_id"]).encode()
                    ).hexdigest(),
                    16,
                )
                % 5
                for r in self.rows
            ]
        )
        controls = [
            "zero",
            "condition",
            "fixed_precursor",
            "calibrated_precursor",
            "sequence_0.001",
            "sequence_0.01",
            "sequence_0.1",
        ]
        self.baselines = {}
        for name in controls:
            self.baselines[name] = {}
            for population in ["train", "inner_cv"]:
                data = pq.read_table(
                    baseline / f"results/fits/{name}-{population}.parquet",
                    filters=[("condition_id", "=", "hap1_erap1_ko")],
                ).to_pydict()
                assert data["measurement_id"] == [r["measurement_id"] for r in self.rows]
                assert np.array_equal(data["target_log10_ratio"], self.target)
                self.baselines[name][population] = np.array(data["prediction_log10_ratio"])
                self.weight = np.array(data["weight"])
        self.weight /= self.weight.sum()
        self.checkpoints = read(baseline / "results/fits/checkpoints.json")["sequence_0.1"]
        self.parent_sequences = {
            p["parent_id"]: p["reference_sequence"]
            for p in pq.read_table(recipe / "derived/parents.parquet").to_pylist()
        }
        references = {
            p["reference_record_id"]: p["sequence_sha256"]
            for p in pq.read_table(
                recipe / "source/references.parquet",
                columns=["reference_record_id", "sequence_sha256"],
            ).to_pylist()
        }
        origins = defaultdict(set)
        for o in pq.read_table(recipe / "source/occurrences.parquet").to_pylist():
            peptide = o["peptide_sequence"]
            if peptide in qi and o["reference_record_id"] is not None:
                parent = references[o["reference_record_id"]]
                start, end = o["start_1based"] - 1, o["end_1based"]
                sequence = self.parent_sequences[parent]
                assert sequence[start:end] == peptide and set(sequence) <= set(AA)
                origins[peptide].add((parent, start, end))
        inputs = {
            (p["parent_id"], p["condition_id"]): p
            for p in pq.read_table(recipe / "derived/precursor_inputs.parquet").to_pylist()
        }
        birth, counts, owner, parent_input, starts, ends, records = [], [], [], [], [], [], []
        cursor = 0
        for peptide in self.peptides:
            starts.append(cursor)
            for parent, a, b in sorted(origins[peptide]):
                sequence = self.parent_sequences[parent]
                parent_value = inputs[parent, "hap1_erap1_ko"]
                assert parent_value["status"] == "complete"
                j = np.arange(a + 1)
                total = b - j
                left = j > 0
                right = b < len(sequence)
                values = 1 / total.astype(float)
                values[left | right] /= total[left | right] + 1
                values[left & right] *= 2 / (total[left & right] + 2)
                prefix = np.zeros((a + 1, 20), dtype=np.int32)
                if a:
                    prefix[1:] = np.cumsum(
                        np.eye(20, dtype=np.int32)[[AA.index(r) for r in sequence[:a]]], axis=0
                    )
                removed = prefix[-1] - prefix
                counts.append(removed.astype(np.int16))
                birth.append(np.log(values))
                owner.append(np.full(a + 1, qi[peptide], dtype=np.int64))
                parent_input.append(np.full(a + 1, parent_value["log10_response"]))
                records.append(
                    {
                        "peptide_sequence": peptide,
                        "query_id": qi[peptide],
                        "parent_id": parent,
                        "start0": a,
                        "end0": b,
                        "path_start": cursor,
                        "path_end": cursor + a + 1,
                        "parent_log10_response": parent_value["log10_response"],
                    }
                )
                cursor += a + 1
            ends.append(cursor)
        self.arrays = {
            "log_birth": np.concatenate(birth),
            "counts": np.concatenate(counts),
            "owner": np.concatenate(owner),
            "input_log10": np.concatenate(parent_input),
            "starts": np.array(starts),
            "ends": np.array(ends),
            "target_residue": np.array([AA.index(p[0]) for p in self.peptides]),
            "target_monomer": np.array([len(p) == 1 for p in self.peptides]),
        }
        assert cursor == 1741916 and len(self.peptides) == 3304 and len(records) == 3409
        self.origins = records
        self.tensor = {
            k: torch.as_tensor(
                v,
                dtype=torch.float64
                if k in {"log_birth", "counts", "input_log10"}
                else torch.bool
                if k == "target_monomer"
                else torch.long,
            )
            for k, v in self.arrays.items()
        }
        self.row_index = torch.tensor(self.owner_rows)
        self.y = torch.tensor(self.target)
        self.x = torch.tensor(self.baseline_input)
        self.w = torch.tensor(self.weight)

    def prior(self, fold):
        return self.checkpoints["full"] if fold == "full" else self.checkpoints["folds"][str(fold)]

    def baseline_for_fit(self, fold):
        # Reconstruct frozen fold coefficients for all rows: inner-training values
        # must not come from a different fold's held-out prediction.
        checkpoint = self.prior(fold)
        gamma = checkpoint["gamma"]
        phi = np.array(
            [
                [
                    len(p),
                    *[p.count(a) / len(p) for a in AA],
                    *[float(p[0] == a) for a in AA],
                    *[float(p[-1] == a) for a in AA],
                ]
                for p in self.peptides
            ]
        )
        mask = np.array(checkpoint["feature_mask"])
        mean, sd = np.array(checkpoint["feature_mean"]), np.array(checkpoint["feature_sd"])
        parameters = checkpoint["sequence"]["hap1_erap1_ko"]
        z = (phi[:, mask] - mean[mask]) / sd[mask] - np.array(parameters["center"])
        sequence = z @ np.array(parameters["beta"])
        prediction = (
            checkpoint["condition_offsets"]["hap1_erap1_ko"]
            + gamma * self.baseline_input
            + sequence[self.owner_rows]
        )
        select = np.ones(len(self.rows), dtype=bool) if fold == "full" else self.fold == fold
        saved = self.baselines["sequence_0.1"]["train" if fold == "full" else "inner_cv"]
        np.testing.assert_allclose(prediction[select], saved[select], rtol=1e-11, atol=1e-11)
        return torch.tensor(prediction), gamma


def physical(dataset, parameter, gamma, *, uniform=False, uniforms=None):
    log_rates = parameter[:1].expand(20) if uniform else parameter[:20]
    z = parameter[-1]
    t = dataset.tensor
    common = [
        t["counts"],
        t["owner"],
        t["starts"],
        t["ends"],
        t["target_residue"],
        t["target_monomer"],
        log_rates,
    ]
    wt = trimmed_log_amount(t["log_birth"], *common, z * 0, uniforms=uniforms)
    ko = trimmed_log_amount(
        t["log_birth"] + math.log(10) * gamma * t["input_log10"],
        *common,
        F.logsigmoid(z),
        uniforms=uniforms,
    )
    return ((ko - wt) / math.log(10))[dataset.row_index]


def objective(dataset, parameter, fold, arm, baseline, gamma, *, uniforms=None, penalty=True):
    selected = (
        torch.ones(len(dataset.rows), dtype=torch.bool)
        if fold == "full"
        else torch.tensor(dataset.fold != fold)
    )
    w = dataset.w[selected] / dataset.w[selected].sum()
    raw = physical(dataset, parameter, gamma, uniform=arm == "uniform_exact", uniforms=uniforms)
    correction = raw - gamma * dataset.x if arm.startswith("augmented") else raw
    uncalibrated = baseline + correction if arm.startswith("augmented") else raw
    offset = (w * (dataset.y[selected] - uncalibrated[selected])).sum()
    prediction = offset + uncalibrated
    data_loss = (w * (prediction[selected] - dataset.y[selected]).square()).sum()
    regularizer = 0.01 * parameter[:-1].square().mean() + 0.01 * parameter[-1].square()
    loss = data_loss + regularizer if penalty else data_loss
    return (
        loss,
        prediction,
        {
            "data_loss": data_loss,
            "regularizer": regularizer,
            "offset": offset,
            "physical": raw,
            "correction": correction,
        },
    )
