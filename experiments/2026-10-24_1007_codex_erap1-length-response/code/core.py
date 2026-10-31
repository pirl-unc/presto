"""Fixed HAP1 consumer plus two remaining-length modifiers; no source curation."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from presto.models.n_terminal_trimming import (
    state_trimmed_log_amount,
    trimming_origin_layout,
)

ROOT = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


CFG = read(ROOT / "reproduce/launch.json")
REFERENCE = Path(CFG["reference_experiment"])
for name, expected in CFG["reference_sha256"].items():
    assert sha(REFERENCE / name) == expected
spec = importlib.util.spec_from_file_location("erap1_reference_core", REFERENCE / "code/core.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


class Dataset(legacy.Dataset):
    def __init__(self):
        super().__init__()
        self.cfg = CFG
        for name, expected in read(REFERENCE / "results/closure.json")["artifact_sha256"].items():
            if name.startswith("results/fits/"):
                assert sha(REFERENCE / name) == expected
        residues, lengths, terminal = [], [], []
        for origin in self.origins:
            a, b = origin["start0"], origin["end0"]
            sequence = self.parent_sequences[origin["parent_id"]]
            residues.append(np.array([AA.index(r) for r in sequence[: a + 1]], dtype=np.int64))
            lengths.append(b - np.arange(a + 1, dtype=np.int64))
            terminal.append(np.full(a + 1, origin["path_end"] - 1, dtype=np.int64))
        self.arrays.update(
            state_residue=np.concatenate(residues),
            state_length=np.concatenate(lengths),
            terminal_index=np.concatenate(terminal),
        )
        length = self.arrays["state_length"]
        self.arrays["state_band"] = np.where(length <= 8, 0, np.where(length <= 16, 1, 2))
        self.arrays["state_monomer"] = length == 1
        for key in [
            "state_residue",
            "state_length",
            "terminal_index",
            "state_band",
            "state_monomer",
        ]:
            self.tensor[key] = torch.as_tensor(self.arrays[key])
        self.layout = trimming_origin_layout(self.tensor["terminal_index"])


def physical(dataset, parameter, gamma, *, uniform=False, uniforms=None):
    n_rates = 1 if uniform else 20
    log_rates = parameter[:1].expand(20) if uniform else parameter[:20]
    eta = torch.stack((parameter[n_rates], parameter[n_rates] * 0, parameter[n_rates + 1]))
    z = parameter[-1]
    t = dataset.tensor
    state_rates = log_rates[t["state_residue"]] + eta[t["state_band"]]
    common = [t["terminal_index"], t["state_monomer"], t["owner"], t["starts"], t["ends"]]
    wt = state_trimmed_log_amount(
        t["log_birth"], state_rates, *common, uniforms=uniforms, origin_layout=dataset.layout
    )
    ko = state_trimmed_log_amount(
        t["log_birth"] + math.log(10) * gamma * t["input_log10"],
        state_rates + F.logsigmoid(z),
        *common,
        uniforms=uniforms,
        origin_layout=dataset.layout,
    )
    return ((ko - wt) / math.log(10))[dataset.row_index]


def objective(dataset, parameter, fold, arm, baseline, gamma, *, uniforms=None, penalty=True):
    selected = (
        torch.ones(len(dataset.rows), dtype=torch.bool)
        if fold == "full"
        else torch.tensor(dataset.fold != fold)
    )
    w = dataset.w[selected] / dataset.w[selected].sum()
    raw = physical(dataset, parameter, gamma, uniform=arm == "length_exact", uniforms=uniforms)
    correction = raw - gamma * dataset.x if arm.startswith("augmented") else raw
    uncalibrated = baseline + correction if arm.startswith("augmented") else raw
    offset = (w * (dataset.y[selected] - uncalibrated[selected])).sum()
    prediction = offset + uncalibrated
    data_loss = (w * (prediction[selected] - dataset.y[selected]).square()).sum()
    regularizer = (
        dataset.cfg["rate_penalty"] * parameter[:-3].square().mean()
        + dataset.cfg["length_penalty"] * parameter[-3:-1].square().mean()
        + dataset.cfg["loss_penalty"] * parameter[-1].square()
    )
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
