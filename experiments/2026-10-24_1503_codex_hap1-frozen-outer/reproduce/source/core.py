"""Frozen covariate-only inference and prespecified evaluation reductions."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from presto.models.n_terminal_trimming import (
    state_trimmed_log_amount,
    trimming_origin_layout,
)
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
INPUT_COLUMNS = ["peptide_sequence", "condition_id", "precursor_log10_ratio"]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def direct(rows, checkpoint):
    """No fit, target, weighting, or evaluation-derived normalization."""
    if checkpoint["variant"] == "zero":
        return np.zeros(len(rows))
    condition = np.array([r["condition_id"] for r in rows])
    result = checkpoint["gamma"] * np.array([r["precursor_log10_ratio"] for r in rows])
    result += np.array([checkpoint["condition_offsets"][c] for c in condition])
    if checkpoint["variant"].startswith("sequence_"):
        phi = np.array(
            [
                [
                    len(p),
                    *[p.count(a) / len(p) for a in AA],
                    *[float(p[0] == a) for a in AA],
                    *[float(p[-1] == a) for a in AA],
                ]
                for p in (r["peptide_sequence"] for r in rows)
            ]
        )
        mask = np.array(checkpoint["feature_mask"])
        mean, sd = np.array(checkpoint["feature_mean"]), np.array(checkpoint["feature_sd"])
        z = (phi[:, mask] - mean[mask]) / sd[mask]
        for c in sorted(set(condition)):
            select = condition == c
            cp = checkpoint["sequence"][c]
            result[select] += (z[select] - cp["center"]) @ np.array(cp["beta"])
    return result


class Source:
    def __init__(self, recipe):
        self.parents = {
            r["parent_id"]: r["reference_sequence"]
            for r in pq.read_table(recipe / "derived/parents.parquet").to_pylist()
        }
        references = {
            r["reference_record_id"]: r["sequence_sha256"]
            for r in pq.read_table(
                recipe / "source/references.parquet",
                columns=["reference_record_id", "sequence_sha256"],
            ).to_pylist()
        }
        self.origins = defaultdict(set)
        for r in pq.read_table(recipe / "source/occurrences.parquet").to_pylist():
            if r["reference_record_id"] is not None:
                self.origins[r["peptide_sequence"]].add(
                    (references[r["reference_record_id"]], r["start_1based"] - 1, r["end_1based"])
                )
        self.inputs = {
            r["parent_id"]: r
            for r in pq.read_table(
                recipe / "derived/precursor_inputs.parquet",
                filters=[("condition_id", "=", "hap1_erap1_ko")],
            ).to_pylist()
        }


class Paths:
    """Observed query paths; only source sequences and precursor covariates enter."""

    def __init__(self, rows, source):
        assert all(r["condition_id"] == "hap1_erap1_ko" for r in rows)
        self.peptides = sorted({r["peptide_sequence"] for r in rows})
        index = {p: i for i, p in enumerate(self.peptides)}
        self.row_index = torch.tensor([index[r["peptide_sequence"]] for r in rows])
        parts = defaultdict(list)
        self.origins = []
        self.parents = {}
        starts, ends, cursor = [], [], 0
        for peptide in self.peptides:
            starts.append(cursor)
            assert source.origins[peptide]
            for parent, a, b in sorted(source.origins[peptide]):
                sequence = source.parents[parent]
                assert sequence[a:b] == peptide and set(sequence) <= set(AA)
                precursor = source.inputs[parent]
                assert precursor["status"] == "complete"
                self.parents[parent] = sequence
                length = b - np.arange(a + 1)
                left, right = np.arange(a + 1) > 0, b < len(sequence)
                birth = 1 / length.astype(float)
                birth[left | right] /= length[left | right] + 1
                birth[left & right] *= 2 / (length[left & right] + 2)
                parts["log_birth"].append(np.log(birth))
                parts["input_log10"].append(np.full(a + 1, precursor["log10_response"]))
                parts["state_residue"].append(np.array([AA.index(v) for v in sequence[: a + 1]]))
                parts["state_band"].append(np.where(length <= 8, 0, np.where(length <= 16, 1, 2)))
                parts["state_monomer"].append(length == 1)
                parts["terminal_index"].append(np.full(a + 1, cursor + a, dtype=np.int64))
                parts["owner"].append(np.full(a + 1, index[peptide], dtype=np.int64))
                self.origins.append(
                    {
                        "peptide_sequence": peptide,
                        "query_id": index[peptide],
                        "parent_id": parent,
                        "start0": a,
                        "end0": b,
                        "path_start": cursor,
                        "path_end": cursor + a + 1,
                        "parent_log10_response": precursor["log10_response"],
                    }
                )
                cursor += a + 1
            ends.append(cursor)
        self.arrays = {k: np.concatenate(v) for k, v in parts.items()}
        self.arrays.update(starts=np.array(starts), ends=np.array(ends))
        self.t = {k: torch.as_tensor(v) for k, v in self.arrays.items()}
        self.layout = trimming_origin_layout(self.t["terminal_index"])
        self.n_paths = cursor

    def physical(self, checkpoint):
        p = torch.tensor(checkpoint["parameter"], dtype=torch.float64)
        uniform = len(p) in (2, 4)
        rates = p[:1].expand(20) if uniform else p[:20]
        eta = p.new_zeros(3)
        if len(p) in (4, 23):
            eta = torch.stack((p[-3], p[-3] * 0, p[-2]))
        t = self.t
        rate = rates[t["state_residue"]] + eta[t["state_band"]]
        args = [t["terminal_index"], t["state_monomer"], t["owner"], t["starts"], t["ends"]]
        with torch.no_grad():
            wt = state_trimmed_log_amount(t["log_birth"], rate, *args, origin_layout=self.layout)
            ko = state_trimmed_log_amount(
                t["log_birth"] + math.log(10) * checkpoint["gamma"] * t["input_log10"],
                rate + F.logsigmoid(p[-1]),
                *args,
                origin_layout=self.layout,
            )
        return ((ko - wt) / math.log(10))[self.row_index].numpy()


def physical_prediction(paths, rows, checkpoint, baseline):
    raw = paths.physical(checkpoint)
    correction = raw.copy()
    if checkpoint["arm"].startswith("augmented"):
        correction -= checkpoint["gamma"] * np.array([r["precursor_log10_ratio"] for r in rows])
        prediction = baseline + correction + checkpoint["offset"]
    else:
        prediction = raw + checkpoint["offset"]
    return prediction, correction


def weights(rows):
    counts = Counter((r["condition_id"], r["component_id"], r["peptide_sequence"]) for r in rows)
    peptides = Counter((c, g) for c, g, _ in counts)
    groups = Counter(c for c, _ in peptides)
    return np.array(
        [
            1
            / (
                len(groups)
                * groups[r["condition_id"]]
                * peptides[r["condition_id"], r["component_id"]]
                * counts[r["condition_id"], r["component_id"], r["peptide_sequence"]]
            )
            for r in rows
        ]
    )


def metrics(y, prediction, w):
    w = w / w.sum()
    error = prediction - y
    mse = float(w @ error**2)
    variance = float(w @ (y - w @ y) ** 2) if np.ptp(y) else 0.0
    varying = len(y) > 1 and np.ptp(prediction) > 1e-14 and np.ptp(y) > 0
    return {
        "n_observations": len(y),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(w @ abs(error)),
        "r2": 1 - mse / variance if variance else None,
        "pearson": float(pearsonr(y, prediction).statistic) if varying else None,
        "spearman": float(spearmanr(y, prediction).statistic) if varying else None,
        "prediction_sd": math.sqrt(float(w @ (prediction - w @ prediction) ** 2))
        if np.ptp(prediction)
        else 0.0,
        "target_sd": math.sqrt(variance),
    }


def strata(rows):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups["condition", r["condition_id"]].append(i)
        groups["condition_component", r["condition_id"] + ":" + r["component_id"]].append(i)
        groups["length", str(len(r["peptide_sequence"]))].append(i)
    return {k: np.array(v) for k, v in groups.items()}
