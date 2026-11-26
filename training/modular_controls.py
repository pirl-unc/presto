"""Opt-in affinity-first training controls for the common modular trainer.

Every option defaults to the historical behavior. Unset options are never written into the
resolved ``training`` dict (`control` reads through `CONTROL_DEFAULTS`), so default manifests keep
their resolved config and resume contract. See tasks/affinity-first-training-spec.md.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math

import torch
from torch.nn import functional as F

from ..data.assay_measurement import observation_method
from ..data.modular import Observation, endpoint_spec, observation_target
from .modular_sampling import validate_stratum_weights

CONTROL_DEFAULTS = {
    "sampling_mode": "epoch",
    "loss_normalization": "observation_mean",
    "endpoint_loss_weights": {},
    "schedule": [],
    "lr_schedule": None,
    "new_module_lr_scale": None,
    "grad_clip_norm": None,
    "ema_decay": None,
    "within_group_ranking_loss": None,
    "state_contrast_loss": None,
    "processing_pair_training": None,
    "apc_contrast_training": None,
    "allele_swap_training": None,
    "allele_diff_training": None,
    "tcell_ranking_weight": 0.0,
    "amp_bfloat16": False,
    "elution_pmhc_residual_l2_weight": 0.0,
    "ms_response_elution_l2_weight": 0.0,
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
    pairs = control(training, "processing_pair_training")
    if pairs is not None:
        _check_options(
            pairs,
            {"weight", "pairs_per_class", "logit_weight", "selection_weight", "margin"},
            "processing_pair_training",
        )
        if not _number(pairs.get("weight")) or pairs["weight"] <= 0:
            raise ValueError("processing_pair_training requires positive finite weight")
        if "logit_weight" in pairs and (
            not _number(pairs["logit_weight"]) or pairs["logit_weight"] < 0
        ):
            raise ValueError("processing_pair_training logit_weight must be nonnegative finite")
        if "selection_weight" in pairs and (
            not _number(pairs["selection_weight"]) or pairs["selection_weight"] < 0
        ):
            raise ValueError("processing_pair_training selection_weight must be nonnegative finite")
        if "margin" in pairs and (not _number(pairs["margin"]) or pairs["margin"] < 0):
            raise ValueError("processing_pair_training margin must be nonnegative finite")
        if type(pairs.get("pairs_per_class")) is not int or pairs["pairs_per_class"] <= 0:
            raise ValueError("processing_pair_training requires positive integer pairs_per_class")
        if (
            config.elution_staging != "affinity_downstream"
            or control(training, "sampling_mode") != "weighted_steps"
        ):
            raise ValueError(
                "processing_pair_training requires staged presentation and weighted_steps"
            )
    if type(control(training, "amp_bfloat16")) is not bool:
        raise ValueError("amp_bfloat16 must be boolean")
    for reg_key in (
        "elution_pmhc_residual_l2_weight",
        "ms_response_elution_l2_weight",
        "tcell_ranking_weight",
    ):
        val = control(training, reg_key)
        if not _number(val) or val < 0:
            raise ValueError(f"{reg_key} must be nonnegative finite")
    allele_swap = control(training, "allele_swap_training")
    if allele_swap is not None:
        _check_options(
            allele_swap,
            {"weight", "anchors_per_class", "negatives_per_anchor", "temperature", "max_buffer"},
            "allele_swap_training",
        )
        if not _number(allele_swap.get("weight")) or allele_swap["weight"] <= 0:
            raise ValueError("allele_swap_training requires positive finite weight")
        for int_k in ("anchors_per_class", "negatives_per_anchor", "max_buffer"):
            if int_k in allele_swap and (
                type(allele_swap[int_k]) is not int or allele_swap[int_k] <= 0
            ):
                raise ValueError(f"allele_swap_training {int_k} must be a positive integer")
        if "temperature" in allele_swap and (
            not _number(allele_swap["temperature"]) or allele_swap["temperature"] <= 0
        ):
            raise ValueError("allele_swap_training temperature must be positive finite")
    allele_diff = control(training, "allele_diff_training")
    if allele_diff is not None:
        _check_options(
            allele_diff,
            {
                "weight",
                "pairs_per_class",
                "hard_close_fraction",
                "beta",
                "min_diff_log10",
                "close_identity_threshold",
                "max_pairs_per_group",
            },
            "allele_diff_training",
        )
        if not _number(allele_diff.get("weight")) or allele_diff["weight"] <= 0:
            raise ValueError("allele_diff_training requires positive finite weight")
        for int_k in ("pairs_per_class", "max_pairs_per_group"):
            if int_k in allele_diff and (
                type(allele_diff[int_k]) is not int or allele_diff[int_k] <= 0
            ):
                raise ValueError(f"allele_diff_training {int_k} must be a positive integer")
        for pos_k in ("beta", "min_diff_log10"):
            if pos_k in allele_diff and (
                not _number(allele_diff[pos_k]) or allele_diff[pos_k] <= 0
            ):
                raise ValueError(f"allele_diff_training {pos_k} must be positive finite")
        for frac_k in ("hard_close_fraction", "close_identity_threshold"):
            if frac_k in allele_diff and (
                not _number(allele_diff[frac_k]) or not 0.0 <= allele_diff[frac_k] <= 1.0
            ):
                raise ValueError(f"allele_diff_training {frac_k} must be in [0, 1]")
    apc_contrast = control(training, "apc_contrast_training")
    if apc_contrast is not None:
        _check_options(
            apc_contrast,
            {
                "weight",
                "selectivity_per_step",
                "yield_per_step",
                "yield_weight",
                "margin",
                "quant_per_step",
                "quant_weight",
                "quant_rank_weight",
                "quant_scale",
            },
            "apc_contrast_training",
        )
        if not _number(apc_contrast.get("weight")) or apc_contrast["weight"] <= 0:
            raise ValueError("apc_contrast_training requires positive finite weight")
        sel_n = apc_contrast.get("selectivity_per_step", 4)
        yld_n = apc_contrast.get("yield_per_step", 4)
        qnt_n = apc_contrast.get("quant_per_step", 0)
        if (
            type(sel_n) is not int
            or sel_n < 0
            or type(yld_n) is not int
            or yld_n < 0
            or type(qnt_n) is not int
            or qnt_n < 0
        ):
            raise ValueError("apc_contrast_training draw counts must be nonnegative integers")
        if sel_n == 0 and yld_n == 0 and qnt_n == 0:
            raise ValueError("apc_contrast_training requires at least one active contrast draw")
        for opt_key in (
            "yield_weight",
            "margin",
            "quant_weight",
            "quant_rank_weight",
            "quant_scale",
        ):
            if opt_key in apc_contrast and (
                not _number(apc_contrast[opt_key]) or apc_contrast[opt_key] < 0
            ):
                raise ValueError(f"apc_contrast_training {opt_key} must be nonnegative finite")
        if config.elution_staging != "affinity_downstream":
            raise ValueError("apc_contrast_training requires staged presentation")
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
    lr_scale = control(training, "new_module_lr_scale")
    if lr_scale is not None and (not _number(lr_scale) or lr_scale <= 0):
        raise ValueError("new_module_lr_scale must be positive and finite or null")
    clip = control(training, "grad_clip_norm")
    if clip is not None and (not _number(clip) or clip <= 0):
        raise ValueError("grad_clip_norm must be positive and finite or null")
    decay = control(training, "ema_decay")
    if decay is not None and (not _number(decay) or not 0 < decay < 1):
        raise ValueError("ema_decay must be in (0, 1) or null")

    contrast = control(training, "state_contrast_loss")
    if contrast is not None:
        _check_options(contrast, {"weight"}, "state_contrast_loss")
        if not _number(contrast.get("weight")) or contrast["weight"] <= 0:
            raise ValueError("state contrast weight must be positive and finite")

    ranking = control(training, "within_group_ranking_loss")
    if ranking is not None:
        _check_options(
            ranking,
            {"weight", "endpoints", "grouping", "include_censored"},
            "within_group_ranking_loss",
        )
        if "include_censored" in ranking and type(ranking["include_censored"]) is not bool:
            raise ValueError("ranking include_censored must be boolean")
        if ranking.get("grouping", "study") not in ("study", "assay_allele"):
            raise ValueError("unsupported ranking grouping")
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
        _check_options(batching, {"rows_per_cell", "grouping"}, "cell_batching")
        if batching.get("grouping", "study") not in ("study", "assay_allele"):
            raise ValueError("unsupported cell batching grouping")
        if ranking and batching.get("grouping", "study") != ranking.get("grouping", "study"):
            raise ValueError("ranking and cell batching grouping must agree")
        rows = batching.get("rows_per_cell")
        if type(rows) is not int or not 2 <= rows <= training["batch_size"]:
            raise ValueError("rows_per_cell must be an integer in [2, batch_size]")

    metric = control(training, "selection_metric")
    if metric not in ("validation_loss", "within_study_spearman", "within_assay_allele_spearman"):
        raise ValueError("unsupported selection_metric")
    names, minimum = (
        control(training, "selection_spearman_endpoints"),
        control(training, "selection_min_cell_rows"),
    )
    if type(minimum) is not int or minimum < 2:
        raise ValueError("selection_min_cell_rows must be an integer >= 2")
    if metric in ("within_study_spearman", "within_assay_allele_spearman"):
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


def cell_key(row, grouping="study"):
    """(endpoint, source, assay method, study) comparison cell; None without a study."""
    if grouping == "assay_allele":
        return assay_allele_key(row)
    if grouping != "study":
        raise ValueError("unsupported comparison grouping")
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


ASSAY_CELL_FIELDS = (
    "endpoint",
    "mhc_class",
    "source",
    "study",
    "preparation",
    "mechanism",
    "detector",
    "mhc_key",
)


def assay_allele_key(row):
    """A resolved single-pMHC assay cell; unknown methods/ambiguous MHCs are excluded."""
    study, method = study_of(row.provenance), observation_method(row)
    if study is None or not method.usable or len(row.mhc) != 1:
        return None
    mhc = row.mhc[0]
    mhc_class = mhc.class_override or mhc.mhc_class
    if mhc_class not in ("I", "II"):
        return None
    # Sequence identity avoids spelling/alias artifacts without conflating alleles.
    mhc_key = hashlib.sha256((mhc.groove1 + "|" + mhc.groove2).encode()).hexdigest()
    return (
        row.endpoint,
        mhc_class,
        row.source,
        str(study),
        method.preparation,
        method.mechanism,
        method.detector,
        mhc_key,
    )


def selection_cell(key, count):
    fields = ASSAY_CELL_FIELDS if len(key) == len(ASSAY_CELL_FIELDS) else ("endpoint", "study")
    return {**dict(zip(fields, key, strict=True)), "n": count}


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


def ranking_loss(rows, outputs, endpoints, *, grouping="study", include_censored=False):
    """RankNet-style pairwise logistic loss within same-cell rows of one batch.

    Returns (mean loss or None, pair count). Only training rows/labels are touched.
    When `include_censored=True`, also pairs exact/left-censored binders against
    strictly weaker right-censored non-binders within the same cell.
    """
    cells = defaultdict(list)
    for row, output in zip(rows, outputs, strict=True):
        if row.endpoint not in endpoints or row.value is None:
            continue
        if not include_censored and row.qualifier != "=":
            continue
        if row.qualifier not in ("=", "<", "<=", ">", ">="):
            continue
        key = cell_key(row, grouping)
        if key is not None:
            cells[key].append((observation_target(row), row.qualifier, output.value))
    losses = []
    for members in cells.values():
        if len(members) < 2:
            continue
        targets = torch.tensor([target for target, _, _ in members])
        scores = torch.stack([score for _, _, score in members])
        if not include_censored:
            lower = targets[:, None] < targets[None, :]
        else:
            n = len(members)
            lower = torch.zeros((n, n), dtype=torch.bool)
            for i in range(n):
                ti, qi, _ = members[i]
                for j in range(n):
                    if i == j:
                        continue
                    tj, qj, _ = members[j]
                    if qi == "=" and qj == "=" and ti < tj:
                        lower[i, j] = True
                    elif qi in ("=", "<", "<=") and qj in (">", ">="):
                        if ti < tj or (ti == tj and (qi == "<" or qj == ">")):
                            lower[i, j] = True
                    elif qi in ("<", "<=") and qj == "=" and ti < tj:
                        lower[i, j] = True
        difference = scores[:, None] - scores[None, :]
        if lower.any():
            losses.append(F.softplus(difference[lower]))
    if not losses:
        return None, 0
    losses = torch.cat(losses)
    return losses.mean(), int(losses.numel())


def tcell_ranking_loss(rows, outputs):
    """Pairwise AUROC ranking loss for binary `tcell_response` observations.

    Scores `output.components["tcell_recognition"]` when staged (track-free biological
    immunogenicity) or `output.value` otherwise, contrasting positive (`1.0`) vs negative
    (`0.0`) rows within each `(mhc_class, track_family)` group (and within-study when present).
    """
    by_family = defaultdict(list)
    by_study = defaultdict(list)
    for row, output in zip(rows, outputs, strict=True):
        if row.endpoint != "tcell_response" or row.value is None:
            continue
        if row.value not in (0.0, 1.0):
            continue
        mhc_cls = (row.mhc[0].class_override or row.mhc[0].mhc_class) if row.mhc else "none"
        score = output.components.get("tcell_recognition", output.value)
        if score is None:
            continue
        fam = row.tcell_track.family if row.tcell_track is not None else "default"
        by_family[(mhc_cls, fam)].append((float(row.value), score))
        study = study_of(row.provenance)
        if study is not None:
            by_study[(mhc_cls, str(study))].append((float(row.value), score))

    losses = []
    for bucket in (*by_study.values(), *by_family.values()):
        pos = [sc for y, sc in bucket if y == 1.0]
        neg = [sc for y, sc in bucket if y == 0.0]
        if not pos or not neg:
            continue
        p_t = torch.stack(pos)
        n_t = torch.stack(neg)
        diff = n_t[:, None] - p_t[None, :]
        losses.append(F.softplus(diff).reshape(-1))
    if not losses:
        return None, 0
    all_pairs = torch.cat(losses)
    return all_pairs.mean(), int(all_pairs.numel())


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


def assay_allele_spearman_score(records, endpoints, minimum, required_classes):
    """Equal-class/endpoint/allele mean of within-assay-study peptide correlations.

    Historical within-study scoring is unchanged. This prospective mode cannot
    gain ranking credit merely from offsets between assays, classes or alleles.
    Constant targets are not evaluable; constant predictions on varying targets
    score zero. Every requested class/endpoint must have an evaluable cell.
    """
    from .modular_evaluation import _correlation, _ranks

    grouped, excluded = defaultdict(list), 0
    for record in records:
        if record["endpoint"] not in endpoints or record["qualifier"] != "=":
            continue
        key = assay_allele_key(Observation.from_dict(record["observation"]))
        if key is None:
            excluded += 1
        else:
            grouped[key].append(record)
    used, skipped = [], []
    alleles = defaultdict(list)
    for key, rows in sorted(grouped.items()):
        cell = selection_cell(key, len(rows))
        targets = [row["target"] for row in rows]
        if len(rows) < minimum or len(set(targets)) < 2:
            skipped.append(
                {**cell, "reason": "too_small" if len(rows) < minimum else "constant_target"}
            )
            continue
        rho = _correlation(_ranks(targets), _ranks([row["prediction"] for row in rows]))
        rho = 0.0 if rho is None else rho
        used.append({**cell, "spearman": rho})
        alleles[(key[1], key[0], key[-1])].append(rho)
    by_endpoint = defaultdict(list)
    for (mhc_class, endpoint, _), values in alleles.items():
        by_endpoint[(mhc_class, endpoint)].append(sum(values) / len(values))
    required = {(cls, endpoint) for cls in required_classes for endpoint in endpoints}
    if required - by_endpoint.keys():
        raise ValueError(
            f"no evaluable assay/allele validation cell for {sorted(required - by_endpoint.keys())}"
        )
    by_class = {
        cls: sum(sum(by_endpoint[(cls, ep)]) / len(by_endpoint[(cls, ep)]) for ep in endpoints)
        / len(endpoints)
        for cls in required_classes
    }
    return -sum(by_class.values()) / len(by_class), {
        "metric": "within_assay_allele_spearman",
        "by_class": by_class,
        "cells": used,
        "skipped": skipped,
        "unresolved_rows": excluded,
    }


def state_contrast_loss(model, rows):
    """Paired counterfactual supervision of the APM state term.

    Rows carry ``provenance['contrast_state']`` ({"interventions": {GENE: "absent"|"present"},
    "cytokines": {name: "present"}}) and ``provenance['contrast_label']`` (1 = observed in the
    perturbed arm, 0 = observed in its same-study control). For each such positive peptide the
    loss is BCE(s(x | contrast state) - s(x | null state), label): the state-induced logit shift
    must separate perturbed-arm peptides from control peptides, which the pair-ranking objective
    cannot see.
    Returns (loss or None, n_rows).
    """
    from dataclasses import replace

    import torch
    from torch.nn import functional as F

    from ..data.modular import ContextAssertion, Exposure

    chosen = [
        row
        for row in rows
        if row.endpoint == "elution_observation"
        and row.value == 1.0
        and "contrast_state" in row.provenance
    ]
    if not chosen or not getattr(model.config, "state_bottleneck", 0):
        return None, 0
    with_state, null, labels = [], [], []
    for row in chosen:
        spec = row.provenance["contrast_state"]
        interventions = {g: ContextAssertion(v) for g, v in spec.get("interventions", {}).items()}
        cytokines = {k: Exposure(v) for k, v in spec.get("cytokines", {}).items()}
        with_state.append(
            replace(
                row,
                context=replace(row.context, interventions=interventions, cytokines=cytokines),
            )
        )
        null.append(replace(row, context=replace(row.context, interventions={}, cytokines={})))
        labels.append(float(row.provenance["contrast_label"]))
    shift = torch.stack(
        [a.value - b.value for a, b in zip(model(with_state), model(null), strict=True)]
    )
    target = torch.tensor(labels, dtype=shift.dtype, device=shift.device)
    return F.binary_cross_entropy_with_logits(shift, target), len(chosen)
