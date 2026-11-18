# Affinity-conditioned presentation training

The canonical matcher, pair sampler, loss and optional source-boundary module
support both MHC classes. External BA predictions select comparable observed and
sampled peptides; they do not supply experimental positive/negative affinity
labels. Use NetMHCpan BA for class I and NetMHCIIpan BA for class II, preserving
predictor version and score-file provenance. EL ranks are not affinity ranks.

## Match complete biological MHC profiles

```python
from presto.data.affinity_matching import (
    AffinityScore,
    MatchingPolicy,
    assess_match,
    candidate_key,
)

policy = MatchingPolicy(
    metric="ba_rank",
    max_gap=0.5,
    profile="all",
    max_log_rank_gap=0.3,
    rank_floor=0.01,
)
# Repeat for every biological MHC candidate, using actual pinned BA outputs.
key = candidate_key(mhc)
positive_scores = {key: AffinityScore(nm=120.0, rank=0.40)}
background_scores = {key: AffinityScore(nm=150.0, rank=0.45)}
assessment = assess_match(positive_scores, background_scores, policy)
```

Ranks use percentage points in `[0, 100]`. Matching requires identical candidate
sets, the same deterministic best candidate and the declared calipers for every
candidate. The default requires an absolute rank difference at most 0.5 points
and an absolute log10 difference of `(rank + 0.01)` at most 0.3. The relative
caliper prevents a flat 0.5-point window from being excessively loose in the
strong-binding tail; the floor handles printed zeros. It is a documented
construction choice, not measured affinity equivalence or an allele assignment.

Historical nM matching remains available through
`MatchingPolicy(metric="log10_nm", max_gap=0.1, profile="winner", max_log_rank_gap=None)`.
A 0.1 log10(nM) caliper means a maximum `10**0.1 ≈ 1.259` nM ratio. It is not
equivalent to a constant percentile-rank difference. Preserve old benchmarks;
changing a matching policy changes the population and requires baseline rescoring.

## Admit auxiliary pairs through the ordinary source index

Keep the original positive observation once. Add the sampled background as a
canonical TRAIN `Observation` with `elution_observation`, unit `binary`, value 0,
and provenance containing:

```text
synthetic: true
processing_pair_positive_id: <original positive observation ID>
affinity_match:
  schema: presto.affinity_match.v1
  predictor: {name: NetMHCpan or NetMHCIIpan, version: <pinned version>}
  source_sha256: <immutable source digest>
  score_sha256: <immutable scoring artifact/manifest digest>
  policy: <serialized MatchingPolicy>
  positive_scores: <every candidate key -> {nm, rank}>
  background_scores: <every candidate key -> {nm, rank}>
```

Retain explicit `source_protein` and `study_id` in provenance. Positive and
background must have different, equal-length peptides from the same protein and
split group, identical biological MHC sets, matched source-context knownness and
identical APC/observation conditions. Source flanks and true terminus values may
differ by peptide position; verify them against the reference during preparation.
The canonical validator checks complete TRAIN pairs and rechecks the declared BA
calipers. It does not reconstruct a reference proteome or independently rerun the
external predictor; those source/score audits belong in the registered adapter
preparation with immutable artifacts.

Screen all generated background identities against observed peptides, held-out
identities and the complete model ancestry before admission. The current experiment
uses only already protected native TRAIN background identities, pooled by exact
protein and length. It retains all ordinary training observations, including those
without an acceptable affinity match or a resolved APC intervention.

## Canonical training controls

Add the auxiliary source to the same manifest and enable:

```json
{
  "training": {
    "sampling_mode": "weighted_steps",
    "processing_pair_training": {"weight": 0.5, "pairs_per_class": 4}
  }
}
```

The options are optional; existing training is unchanged when absent. Pair
training requires an unfrozen model with
`elution_staging="affinity_downstream"` and complete pairs for both classes.
The sampler gives equal pair counts to I/II and square-root study mass within
each class. Each pair's members co-occur. Auxiliary rows are excluded from
ordinary replay and ordinary TRAIN diagnostic evaluation. The optimizer, ordinary
losses and measured-affinity replay remain active. Pair IDs and sampler RNG are
saved in canonical resume state and `processing-pair-draws.json`.

The added loss is the mean `softplus(-(supply_positive - supply_background))`
times the declared weight. `presto.training.processing_pairs.processing_supply`
extracts the differentiable supply term, including APC supply and the optional
boundary residual. It excludes the binding, register-selection and MS-response
terms. It is a conditional processing proxy, not a physical rate or an independently
identified biological component.

## Optional boundary representation

Set `elution_junction_processing=true` with source context and the staged model.
If an APC state is allocated it must use the harmonized APC representation.
Warm-start expansion requires explicit
`pretrained.allow_junction_expansion=true`. The added readout starts at zero;
existing weights and predictions are preserved at initialization. A checkpoint
with a learned boundary module cannot silently be loaded into a configuration
that discards it.

The module shares one MLP across classes and N/C boundaries. It reads six ordered
residue states on each side of the boundary, presence masks, side/class/terminus
flags and the shared APC state. A missing source flank yields zero correction;
a known empty flank at a true protein end remains a valid input. N/C corrections
add to processing supply before the actual presentation aggregation. Intrinsic
affinity receives no source-flank/APC input, although shared encoder updates during
joint training may change affinity and therefore require held-out retention checks.

Public inference remains `ModularPredictor.predict`. Candidate/register diagnostics
under `elution_terms` expose `binding`, `processing`, APC selection/state and, when
enabled, `junction_processing`. These terms are already included in the prediction;
do not add them to the returned logit again. External NetMHC programs are not
inference dependencies.

## Evidence and limits

The registered implementation experiment is
`experiments/2026-11-17_2330_codex_affinity-conditioned-processing/`. Its source
adapter uses pinned Hitlist 2.1.9 / 43e22662 and GENCODE v39. It admits 3,777 class-I
and 9,114 class-II auxiliary pairs across 76/28 studies, alongside the unchanged
209,695/178,179 native ordinary positive/background pairs. The benchmark and model
decision are recorded there; software support alone is not evidence of predictive
improvement.

Report pooled AUROC and complete-pair wins for both classes, residual BA-only
discrimination, ordinary presentation and measured affinity, paired uncertainty,
and component/input controls. Approximate matching retains residual affinity
differences. Same-protein/condition comparisons cannot validate between-protein
expression or absolute intervention yields. The current hybrid-rank validation
set is monoallelic and contains only four I/three II studies; it does not establish
multi-allelic or held-out-study generalization. TEST outcomes remain sealed during
this development family.

The first full comparison is closed: matched supply training improves pooled affinity-matched AUROC in both classes, but class-II paired ordering remains uncertain. Both selected boundary replicates pass the operational retention guards; the previous general-purpose checkpoint remains the default and the new candidate is explicitly provisional. See the experiment DECISION.md for complete paired intervals and controls. The next live Hitlist loader floor is 2.1.12; this does not rewrite frozen 2.1.9 experiment inputs.
