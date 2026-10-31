"""Opt-in affinity-first training controls for the common modular trainer.

Every option defaults to the historical behavior. Unset options are never written into the
resolved ``training`` dict (`control` reads through `CONTROL_DEFAULTS`), so default manifests keep
their resolved config and resume contract. See tasks/affinity-first-training-spec.md.
"""

from __future__ import annotations

from collections import defaultdict
import math

import torch
from torch.nn import functional as F

from ..data.assay_measurement import observation_method
from ..data.modular import endpoint_spec, observation_target
from .modular_sampling import validate_stratum_weights

CONTROL_DEFAULTS = {
    "loss_normalization": "observation_mean",
    "endpoint_loss_weights": {},
    "schedule": [],
    "lr_schedule": None,
    "grad_clip_norm": None,
    "ema_decay": None,
    "within_group_ranking_loss": None,
    "cell_batching": None,
    "selection_metric": "validation_loss",
    "selection_spearman_endpoints": [],
    "selection_min_cell_rows": 20,
}


def control(training, key):
    return training.get(key, CONTROL_DEFAULTS[key])


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _check_options(options, allowed, name):
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError(f"{name} must be an object with only {sorted(allowed)}")


def validate_controls(training, config, balance_by):
    """Reject malformed or contradictory options; mirrors `preflight` strictness."""
    if control(training, "loss_normalization") not in ("observation_mean", "per_endpoint"):
        raise ValueError("unsupported loss_normalization")
    weights = control(training, "endpoint_loss_weights")
    if not isinstance(weights, dict):
        raise ValueError("endpoint_loss_weights must be an object")
    if weights and control(training, "loss_normalization") != "per_endpoint":
        raise ValueError("endpoint_loss_weights requires loss_normalization per_endpoint")
    for name, weight in weights.items():
        if name not in config.endpoints or not _number(weight) or weight <= 0:
            raise ValueError("endpoint_loss_weights need active endpoints and positive weights")

    schedule = control(training, "schedule")
    if not isinstance(schedule, list):
        raise ValueError("schedule must be a list")
    if schedule and not balance_by:
        raise ValueError("schedule requires balance_by fields")
    previous = 0
    for segment in schedule:
        _check_options(segment, {"until_step", "stratum_weights"}, "schedule segment")
        until = segment.get("until_step")
        if type(until) is not int or until <= previous or "stratum_weights" not in segment:
            raise ValueError("schedule needs increasing positive integer until_step values")
        previous = until
        if not validate_stratum_weights(balance_by, segment["stratum_weights"]):
            raise ValueError("schedule segments need explicit stratum_weights")

    lr = control(training, "lr_schedule")
    if lr is not None:
        _check_options(
            lr, {"warmup_steps", "decay", "min_lr_fraction", "decay_steps"}, "lr_schedule"
        )
        warmup, decay = lr.get("warmup_steps", 0), lr.get("decay", "none")
        fraction, span = lr.get("min_lr_fraction", 0.0), lr.get("decay_steps")
        if type(warmup) is not int or warmup < 0 or decay not in ("none", "cosine"):
            raise ValueError("lr_schedule needs integer warmup_steps >= 0 and decay none|cosine")
        if not _number(fraction) or not 0 <= fraction <= 1:
            raise ValueError("min_lr_fraction must be in [0, 1]")
        if span is not None and (type(span) is not int or span <= warmup):
            raise ValueError("decay_steps must be an integer greater than warmup_steps")
        if decay == "cosine" and span is None and training["max_steps"] is None:
            raise ValueError("cosine decay needs decay_steps or max_steps")
        if decay == "cosine" and span is None and training["max_steps"] <= warmup:
            raise ValueError("max_steps must exceed warmup_steps for cosine decay")
    clip = control(training, "grad_clip_norm")
    if clip is not None and (not _number(clip) or clip <= 0):
        raise ValueError("grad_clip_norm must be positive and finite or null")
    decay = control(training, "ema_decay")
    if decay is not None and (not _number(decay) or not 0 < decay < 1):
        raise ValueError("ema_decay must be in (0, 1) or null")

    ranking = control(training, "within_group_ranking_loss")
    if ranking is not None:
        _check_options(ranking, {"weight", "endpoints"}, "within_group_ranking_loss")
        endpoints = ranking.get("endpoints")
        if not _number(ranking.get("weight")) or ranking["weight"] <= 0:
            raise ValueError("ranking weight must be positive and finite")
        if not isinstance(endpoints, list) or not endpoints:
            raise ValueError("ranking endpoints must be a non-empty list")
        for name in endpoints:
            if name not in config.endpoints or endpoint_spec(name).binary:
                raise ValueError("ranking endpoints must be active non-binary assay endpoints")
    batching = control(training, "cell_batching")
    if batching is not None:
        _check_options(batching, {"rows_per_cell"}, "cell_batching")
        rows = batching.get("rows_per_cell")
        if type(rows) is not int or not 2 <= rows <= training["batch_size"]:
            raise ValueError("rows_per_cell must be an integer in [2, batch_size]")

    metric = control(training, "selection_metric")
    if metric not in ("validation_loss", "within_study_spearman"):
        raise ValueError("unsupported selection_metric")
    names, minimum = (
        control(training, "selection_spearman_endpoints"),
        control(training, "selection_min_cell_rows"),
    )
    if type(minimum) is not int or minimum < 2:
        raise ValueError("selection_min_cell_rows must be an integer >= 2")
    if metric == "within_study_spearman":
        if (
            not isinstance(names, list)
            or not names
            or len(set(names)) != len(names)
            or any(name not in config.endpoints or endpoint_spec(name).binary for name in names)
        ):
            raise ValueError("selection_spearman_endpoints must name distinct affinity endpoints")
        if training["selection_endpoint"] is not None or (
            training["selection_reduction"] != "observation_mean"
        ):
            raise ValueError("selection_metric cannot be combined with other selection overrides")
    elif names:
        raise ValueError("selection_spearman_endpoints requires selection_metric")


def study_of(provenance):
    """Source-defined study identity; None when the row does not declare one."""
    raw = provenance.get("raw_record", {})
    return provenance.get("study_id") or raw.get("pmid") or raw.get("reference_iri") or None


def cell_key(row):
    """(endpoint, source, assay method, study) comparison cell; None without a study."""
    study = study_of(row.provenance)
    if study is None:
        return None
    method = observation_method(row)
    return (
        row.endpoint,
        row.source,
        method.preparation,
        method.mechanism,
        method.detector,
        str(study),
    )


def cell_blocks(order, keys, size, rng):
    """Regroup an epoch order into shuffled blocks of up to `size` same-cell rows.

    The multiset of rows (hence stratum frequencies and epoch length) is unchanged.
    Rows without a cell share one pool, so they are blocked but never form ranking pairs.
    """
    pools = defaultdict(list)
    for i in order:
        pools[keys[i]].append(i)
    blocks = [
        pool[start : start + size]
        for key in sorted(pools, key=repr)
        for pool in (pools[key],)
        for start in range(0, len(pool), size)
    ]
    rng.shuffle(blocks)
    return [i for block in blocks for i in block]


def segment_for(schedule, steps):
    """Index of the active schedule segment; len(schedule) means the top-level weights."""
    for position, segment in enumerate(schedule):
        if steps < segment["until_step"]:
            return position
    return len(schedule)


def learning_rate(base, schedule, steps, max_steps):
    """Learning rate for the update that follows `steps` completed updates."""
    if schedule is None:
        return base
    warmup = schedule.get("warmup_steps", 0)
    if steps < warmup:
        return base * (steps + 1) / (warmup + 1)
    if schedule.get("decay", "none") == "none":
        return base
    span = schedule.get("decay_steps") or max_steps
    floor = schedule.get("min_lr_fraction", 0.0)
    progress = min(1.0, (steps - warmup) / (span - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


class EMA:
    """Exponential moving average of trainable parameters; raw weights stay authoritative."""

    def __init__(self, model, decay, saved=None):
        self.decay = decay
        self.names = [name for name, value in model.named_parameters() if value.requires_grad]
        parameters = dict(model.named_parameters())
        if saved is None:
            self.updates = 0
            self.shadow = {name: parameters[name].detach().clone() for name in self.names}
        else:
            if list(saved["shadow"]) != self.names:
                raise ValueError("resume EMA parameter set changed")
            self.updates = saved["updates"]
            self.shadow = {
                name: saved["shadow"][name].to(parameters[name].device) for name in self.names
            }

    def state(self):
        return {"updates": self.updates, "shadow": self.shadow}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # Standard bias-corrected warmup so early averages do not retain the random init.
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        parameters = dict(model.named_parameters())
        for name in self.names:
            self.shadow[name].mul_(decay).add_(parameters[name].detach(), alpha=1 - decay)

    def swap(self, model):
        return _EMAWeights(self, model)


class _EMAWeights:
    def __init__(self, ema, model):
        self.ema, self.model = ema, model

    @torch.no_grad()
    def __enter__(self):
        parameters = dict(self.model.named_parameters())
        self.raw = {name: parameters[name].detach().clone() for name in self.ema.names}
        for name in self.ema.names:
            parameters[name].copy_(self.ema.shadow[name])

    @torch.no_grad()
    def __exit__(self, *exc):
        parameters = dict(self.model.named_parameters())
        for name in self.ema.names:
            parameters[name].copy_(self.raw[name])


def ranking_loss(rows, outputs, endpoints):
    """RankNet-style pairwise logistic loss within same-cell exact rows of one batch.

    Returns (mean loss or None, pair count). Only training rows/labels are touched.
    """
    cells = defaultdict(list)
    for row, output in zip(rows, outputs, strict=True):
        if row.endpoint in endpoints and row.qualifier == "=" and row.value is not None:
            key = cell_key(row)
            if key is not None:
                cells[key].append((observation_target(row), output.value))
    losses = []
    for members in cells.values():
        if len(members) < 2:
            continue
        targets = torch.tensor([target for target, _ in members])
        scores = torch.stack([score for _, score in members])
        lower = targets[:, None] < targets[None, :]
        difference = scores[:, None] - scores[None, :]
        if lower.any():
            losses.append(F.softplus(difference[lower]))
    if not losses:
        return None, 0
    losses = torch.cat(losses)
    return losses.mean(), int(losses.numel())


def spearman_cells(records, endpoints, minimum):
    """Within-study validation Spearman cells from evaluation records (exact rows only).

    Cells are (endpoint, study). Studies with fewer than `minimum` exact rows are listed as
    skipped; constant predictions have no ranking information and score 0.
    """
    from .modular_evaluation import _correlation, _ranks  # evaluation imports training.modular

    grouped = defaultdict(list)
    for record in records:
        if (
            record["endpoint"] in endpoints
            and "identity_task" not in record
            and record["qualifier"] == "="
        ):
            study = study_of(record["observation"].get("provenance", {}))
            if study is not None:
                grouped[(record["endpoint"], str(study))].append(record)
    used, skipped = [], []
    for (endpoint, study), rows in sorted(grouped.items()):
        cell = {"endpoint": endpoint, "study": study, "n": len(rows)}
        if len(rows) < minimum:
            skipped.append(cell)
            continue
        correlation = _correlation(
            _ranks([row["target"] for row in rows]), _ranks([row["prediction"] for row in rows])
        )
        used.append({**cell, "spearman": 0.0 if correlation is None else correlation})
    return used, skipped


def spearman_score(records, endpoints, minimum):
    """Lower is better: negative mean within-study Spearman, with the cells that entered."""
    used, skipped = spearman_cells(records, endpoints, minimum)
    if not used:
        raise ValueError("no validation cell meets selection_min_cell_rows")
    score = -sum(cell["spearman"] for cell in used) / len(used)
    return score, {"metric": "within_study_spearman", "cells": used, "skipped": skipped}
