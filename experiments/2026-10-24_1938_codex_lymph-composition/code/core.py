"""Full-parent physical amounts and conditional observed-feature objectives."""

import hashlib
import importlib.util
import json
import math
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
LN10 = math.log(10)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


OBSERVER = module(
    REPO / "experiments/2026-10-24_1702_codex_ms1-grouped-observer/code/model.py",
    "frozen_observer",
)
PHYSICAL = module(REPO / "models/steady_processing.py", "physical_amounts")


class Data:
    def __init__(self, block):
        cfg = read(ROOT / "reproduce/launch.json")
        admission = read(ROOT / "results/independent-admission.json")
        assert admission["status"] == "passed"
        self.directory = Path(admission["prepared_directory"]) / block
        self.rows = read(self.directory / "covariates.json")
        self.parents = read(self.directory / "parents.json")
        for name in ("covariates.json", "parents.json"):
            assert sha(self.directory / name) == admission["files"][block + "/" + name]
        self.block = block
        self.raw = OBSERVER.features(self.rows)
        self.split = {
            s: np.array([r["split"] == s for r in self.rows])
            for s in ("train", "val", "test")
        }
        self.group_names = sorted({r["group_id"] for r in self.rows})
        self.group_lookup = {g: i for i, g in enumerate(self.group_names)}
        self.group = np.array([self.group_lookup[r["group_id"]] for r in self.rows])
        self.counts = np.bincount(self.group).astype(float)
        self.indices = [
            np.flatnonzero(self.group == i) for i in range(len(self.group_names))
        ]
        self.group_split = [self.rows[i[0]]["split"] for i in self.indices]
        self.group_component = [self.rows[i[0]]["component_id"] for i in self.indices]
        self.weights = {}
        self.group_weights = {}
        for split in self.split:
            components = {r["component_id"] for r in self.rows if r["split"] == split}
            groups_per_component = Counter(
                c for c, s in zip(self.group_component, self.group_split) if s == split
            )
            gw = np.array(
                [
                    1 / (len(components) * groups_per_component[c]) if s == split else 0
                    for c, s in zip(self.group_component, self.group_split)
                ]
            )
            self.group_weights[split] = gw
            self.weights[split] = gw[self.group] / self.counts[self.group]
            assert np.isclose(self.weights[split].sum(), 1)
        observer = Path(cfg["observer_directory"]) / "results/fit"
        trans = np.load(observer / "transforms.npz")
        state = np.load(observer / "sequence_0.1.npz")
        self.observer = (self.raw - trans["means"]) / trans["scales"]
        self.observer = self.observer[:, state["mask"]] @ state["beta"]
        self.observer = self.center(self.observer)
        offsets = {}
        bond = []
        cursor = 0
        for key, sequence in sorted(self.parents.items()):
            offsets[key] = cursor
            for left, right in pairwise(sequence):
                v = np.zeros(38)
                for off, aa in ((0, left), (19, right)):
                    k = AA.index(aa)
                    if k == 19:
                        v[off : off + 19] = -1
                    else:
                        v[off + k] = 1
                bond.append(v)
                cursor += 1
        self.bond = torch.tensor(np.array(bond), dtype=torch.float64)
        left = []
        right = []
        has_left = []
        has_right = []
        first = []
        last = []
        owner = []
        for i, row in enumerate(self.rows):
            offset = offsets[row["parent_id"]]
            length = len(self.parents[row["parent_id"]])
            for start, end in row["occurrences_0based_halfopen"]:
                assert self.parents[row["parent_id"]][start:end] == row["sequence"]
                first.append(offset + start)
                last.append(offset + end - 1)
                left.append(offset + start - 1 if start > 0 else 0)
                right.append(offset + end - 1 if end < length else 0)
                has_left.append(start > 0)
                has_right.append(end < length)
                owner.append(i)
        for name, values in [
            ("left", left),
            ("right", right),
            ("first", first),
            ("last", last),
            ("owner", owner),
            ("group", self.group),
        ]:
            setattr(self, "t_" + name, torch.tensor(values, dtype=torch.long))
        self.t_has_left = torch.tensor(has_left, dtype=torch.float64)
        self.t_has_right = torch.tensor(has_right, dtype=torch.float64)
        self.t_counts = torch.tensor(self.counts, dtype=torch.float64)
        self.t_observer = torch.tensor(self.observer, dtype=torch.float64)
        self.t_indices = [torch.tensor(i) for i in self.indices]

    def labels(self, split):
        receipt = read(ROOT / "results/independent-admission.json")
        path = self.directory / (split + "-labels.json")
        assert sha(path) == receipt["files"][self.block + "/" + path.name]
        labels = read(path)
        y = np.array(
            [
                labels[r["measurement_id"]]["target_centered_log10_area"]
                if r["split"] == split
                else 0
                for r in self.rows
            ]
        )
        p = np.array(
            [
                labels[r["measurement_id"]]["target_area_fraction"]
                if r["split"] == split
                else 0
                for r in self.rows
            ]
        )
        return y, p

    def center(self, values):
        return OBSERVER.center_numpy(values, self.group, self.counts)

    def center_torch(self, values):
        return OBSERVER.center_torch(values, self.t_group, self.t_counts)

    def score(self, theta, sequence=True, observer=True):
        logrates = theta[0] + (
            self.bond @ theta[1:]
            if sequence
            else torch.zeros(len(self.bond), dtype=torch.float64)
        )
        rates = logrates.exp()
        prefix = torch.cat(
            (torch.zeros(1, dtype=torch.float64), torch.cumsum(rates, 0))
        )
        internal = prefix[self.t_last] - prefix[self.t_first]
        left = rates[self.t_left] * self.t_has_left
        right = rates[self.t_right] * self.t_has_right
        logs = PHYSICAL.log_steady_fragment_amount(
            internal, left, right, torch.tensor(1.0, dtype=torch.float64)
        )
        amounts = torch.zeros(len(self.rows), dtype=torch.float64).index_add(
            0, self.t_owner, logs.exp()
        )
        value = amounts.log() / LN10
        if observer:
            value = value + self.t_observer
        return self.center_torch(value)

    def loss(self, score, target, fractions, kind="mse", generator=None, samples=64):
        if kind == "mse":
            return (
                torch.tensor(self.weights["train"]) * (score - target).square()
            ).sum()
        terms = []
        for index, split, weight in zip(
            self.t_indices, self.group_split, self.group_weights["train"]
        ):
            if split != "train":
                continue
            logq = torch.log_softmax(score[index] * LN10, 0)
            reward = logq - fractions[index].log()
            if kind == "exact":
                value = (logq.exp() * reward).sum()
            elif kind == "score_expectation":
                value = (logq.exp().detach() * reward.detach() * logq).sum()
            else:
                drawn = torch.multinomial(
                    logq.exp().detach(), samples, replacement=True, generator=generator
                )
                r = reward[drawn].detach()
                advantage = (r - r.mean()) * (samples / (samples - 1))
                value = (advantage * logq[drawn]).mean()
            terms.append(weight * value)
        return torch.stack(terms).sum()

    def metrics(self, prediction, split, labels=None):
        y, p = self.labels(split) if labels is None else labels
        mask = self.split[split]
        w = self.weights[split][mask]
        result = OBSERVER.metrics(y[mask], prediction[mask], w)
        totals = {"reverse_kl": 0.0, "forward_kl": 0.0, "total_variation": 0.0}
        for indices, s, weight in zip(
            self.indices, self.group_split, self.group_weights[split]
        ):
            if s != split:
                continue
            logq = prediction[indices] * LN10
            logq = logq - np.logaddexp.reduce(logq)
            q = np.exp(logq)
            logp = np.log(p[indices])
            totals["reverse_kl"] += weight * float(q @ (logq - logp))
            totals["forward_kl"] += weight * float(p[indices] @ (logp - logq))
            totals["total_variation"] += weight * float(abs(q - p[indices]).sum() / 2)
        return {
            **result,
            **totals,
            "n_components": len(
                {r["component_id"] for r in self.rows if r["split"] == split}
            ),
            "n_groups": sum(s == split for s in self.group_split),
        }
