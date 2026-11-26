"""Conditional processing supervision from complete, affinity-balanced TRAIN pairs.

Auxiliary backgrounds refer to existing positive observations; they never enter
ordinary replay or acquire measured-negative status. All model parameters remain
eligible for the ordinary measured-affinity and presentation losses.
"""

from collections import defaultdict
from dataclasses import replace
from itertools import accumulate
import math

import torch
from torch.nn import functional as F

from ..data.affinity_matching import AffinityScore, MatchingPolicy, assess_match, candidate_key


PAIR_KEY = "processing_pair_positive_id"


def validate_pair(positive, background):
    if any(r.split != "train" for r in (positive, background)):
        raise ValueError("processing pairs must contain TRAIN observations only")
    if any(
        r.endpoint != "elution_observation" or r.unit != "binary" or r.qualifier != "="
        for r in (positive, background)
    ) or (positive.value, background.value) != (1, 0):
        raise ValueError("processing pairs require observed-positive/sampled-background labels")
    p = background.provenance
    if p.get(PAIR_KEY) != positive.observation_id or not p.get("synthetic"):
        raise ValueError("processing background needs synthetic status and original positive ID")
    if positive.peptide == background.peptide or len(positive.peptide) != len(background.peptide):
        raise ValueError("processing pair peptides must differ and have equal length")
    if (
        not positive.group_id
        or positive.group_id != background.group_id
        or not p.get("source_protein")
        or p["source_protein"] != positive.provenance.get("source_protein")
    ):
        raise ValueError("processing pairs require identical source protein and split group")
    if positive.provenance.get("study_id") != p.get("study_id") or not p.get("study_id"):
        raise ValueError("processing pairs require identical explicit study")
    keys = {candidate_key(m) for m in positive.mhc}
    if keys != {candidate_key(m) for m in background.mhc} or not keys:
        raise ValueError("processing pairs require identical biological MHC candidates")
    classes = {m.class_override or m.mhc_class for m in positive.mhc}
    if len(classes) != 1:
        raise ValueError("processing pair must have one resolved MHC class")
    for name in ("source_n", "source_c", "n_terminus", "c_terminus"):
        if (getattr(positive.context, name) is None) != (getattr(background.context, name) is None):
            raise ValueError("processing pairs must match source-context knownness")

    def condition(r):
        return replace(
            r.context,
            source_n=None,
            source_c=None,
            n_terminus=None,
            c_terminus=None,
            candidate_junctions=(),
            provenance={},
        )

    if condition(positive) != condition(background):
        raise ValueError("processing pairs require identical APC and observation conditions")
    record = p.get("affinity_match", {})
    if record.get("schema") != "presto.affinity_match.v1":
        raise ValueError("versioned affinity-match provenance required")
    cls = next(iter(classes))
    predictor = record.get("predictor", {})
    allowed_predictors = ("NetMHCpan", "Presto") if cls == "I" else ("NetMHCIIpan", "Presto")
    if predictor.get("name") not in allowed_predictors or not predictor.get("version"):
        raise ValueError("class-appropriate BA predictor/version provenance required")
    if not record.get("source_sha256") or not record.get("score_sha256"):
        raise ValueError("immutable source and score lineage required")
    policy = MatchingPolicy(**record["policy"])
    first = {k: AffinityScore(**v) for k, v in record["positive_scores"].items()}
    second = {k: AffinityScore(**v) for k, v in record["background_scores"].items()}
    if (
        set(first) != keys
        or set(second) != keys
        or not assess_match(first, second, policy).accepted
    ):
        raise ValueError("processing pair fails declared allele-resolved affinity balance")
    return cls


def processing_supply(output):
    """Differentiable supply diagnostic, excluding binding, selection and MS terms."""
    candidates = output.diagnostics.get("candidates", ())
    if not candidates:
        raise ValueError("processing supervision requires staged presentation diagnostics")
    values = []
    for candidate in candidates:
        terms = [
            r["class_probability"] * r["elution_terms"]["processing"]
            for r in candidate["registers"]
        ]
        values.append(torch.stack(terms).sum())
    return torch.stack(values).mean()


def processing_selection(output):
    """Detached-affinity-pooled shared selection term (zero gradient into binding/MS)."""
    candidates = output.diagnostics.get("candidates", ())
    if not candidates:
        raise ValueError("processing supervision requires staged presentation diagnostics")
    values = []
    for candidate in candidates:
        terms = [
            r["class_probability"]
            * r["elution_terms"]
            .get("shared_processing", {})
            .get("selection_detached_pool", r["elution_terms"]["processing"].new_zeros(()))
            for r in candidate["registers"]
        ]
        values.append(torch.stack(terms).sum())
    return torch.stack(values).mean()


def pair_loss(
    outputs,
    *,
    logit_weight: float = 0.0,
    selection_weight: float = 0.0,
    margin: float = 0.0,
):
    if not outputs or len(outputs) % 2:
        raise ValueError("processing loss needs complete ordered positive/background pairs")
    if selection_weight:
        scores = [
            processing_supply(o) + selection_weight * processing_selection(o) for o in outputs
        ]
    else:
        scores = [processing_supply(o) for o in outputs]
    if margin > 0.0:
        supply = torch.stack(
            [
                F.relu(margin - (scores[i] - scores[i + 1]))
                + 0.10 * F.softplus(margin - (scores[i] - scores[i + 1]), beta=3.0)
                for i in range(0, len(scores), 2)
            ]
        ).mean()
    else:
        supply = torch.stack(
            [F.softplus(-(scores[i] - scores[i + 1])) for i in range(0, len(scores), 2)]
        ).mean()
    if not logit_weight:
        return supply
    logits = torch.stack(
        [F.softplus(-(outputs[i].value - outputs[i + 1].value)) for i in range(0, len(outputs), 2)]
    ).mean()
    return supply + logit_weight * logits


class PairPool:
    """Class-balanced pairs; square-root study mass; checkpoint-owned RNG draws."""

    def __init__(self, index):
        locations, auxiliary = {}, []
        for i, row in enumerate(index):
            if row.split == "train":
                locations[row.observation_id] = i
            if PAIR_KEY in row.provenance:
                if row.split != "train":
                    raise ValueError("auxiliary processing data must be TRAIN only")
                auxiliary.append((i, row.provenance[PAIR_KEY]))
        grouped = defaultdict(lambda: defaultdict(list))
        self.auxiliary_indices = {i for i, _ in auxiliary}
        for i, parent in auxiliary:
            if parent not in locations or locations[parent] in self.auxiliary_indices:
                raise ValueError("processing background references absent/non-original positive")
            positive, background = index[locations[parent]], index[i]
            cls = validate_pair(positive, background)
            grouped[cls][str(positive.provenance["study_id"])].append((locations[parent], i))
        if set(grouped) != {"I", "II"}:
            raise ValueError("shared processing training requires complete pairs for both classes")
        self.groups, self.studies, self.cumulative = grouped, {}, {}
        for cls in sorted(grouped):
            self.studies[cls] = sorted(grouped[cls])
            self.cumulative[cls] = list(
                accumulate(math.sqrt(len(grouped[cls][s])) for s in self.studies[cls])
            )
        self.counts = {
            cls: {"pairs": sum(map(len, studies.values())), "studies": len(studies)}
            for cls, studies in grouped.items()
        }

    def draw(self, index, rng, pairs_per_class):
        pairs = []
        for cls in ("I", "II"):
            studies = rng.choices(
                self.studies[cls], cum_weights=self.cumulative[cls], k=pairs_per_class
            )
            for study in studies:
                a, b = rng.choice(self.groups[cls][study])
                pairs.append((index[a], index[b]))
        return pairs
