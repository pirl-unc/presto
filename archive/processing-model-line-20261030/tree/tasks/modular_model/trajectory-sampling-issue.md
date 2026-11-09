Full repository design: [Processing distributions and MS observation design](https://github.com/pirl-unc/presto/blob/23558a14f72c2db4f980992a6bbddf68b765db9a/docs/processing_model_design.md).

# Processing distributions and MS observation design

Proposed extension to [issue 130](https://github.com/pirl-unc/presto/issues/130),
[issue 132](https://github.com/pirl-unc/presto/issues/132) and the
[modular model](https://github.com/pirl-unc/presto/blob/23558a14f72c2db4f980992a6bbddf68b765db9a/docs/modular_model.md) in draft PR 131, October 21, 2026.

**Start with a differentiable model over discrete peptide spans.** Predict
nonnegative weights for processing products, including short fragments and
nonbinders, condition those weights on APC machinery, and connect them to pMHC
selection and an independently anchored MS observation model. Train by ordinary
backpropagation through sums over alternatives. Sample actual peptides and paths
from the fitted model when sampling is useful.

RL is an optional approximation or sampling method later. It is not required
just because cleavage positions, transport choices and peptide sequences are
discrete. The first model should have a small number of trainable components
whose outputs can be checked against data. Detailed intracellular trajectories
need additional evidence beyond the final eluted peptide list.

## Information flow

```text
Source proteins and source amounts
                │
                ▼
Sequence representations and source-coordinate spans
                │
APC species ────┤
Cell state ─────┤ → conditional processing-product weights
Machinery ──────┤        │
Delivery ───────┘        ├→ product-pool observations and sampling
                        │   including short fragments and nonbinders
                        ▼
                 effective selection and retention
                        ▲                 │
Peptide + MHC ─→ pMHC representation       │
                 ├→ affinity heads        ▼
                 └→ stability heads   pMHC amounts by molecule
                                          │
                                          ├→ optional paired-TCR scorer
                                          ▼
                                  recovery and MS observation
                                          ▲
Peptide + acquisition ─→ shared MS response┘
                               ▲
Protein + laboratory digest ─→ bulk peptide supply ─→ bulk MS observations
```

Species/class auxiliary predictions and the existing resolver come before the
affected components. Their representations remain available downstream, and
explicit overrides retain their provenance. Biological defaults never become
training labels. No TCR input is needed to predict a peptide pool or pMHC pool.

## Predict amounts before normalizing a distribution

A span identifies an exact substring of an existing protein, its occurrence and
its source coordinates. A product state adds its compartment/stage and, where
relevant, MHC occupancy. Distinguish an unbound nonbinder, an unoccupied binder,
a bound ligand, unknown compatibility and material removed from the represented
pool. An unbound state need not be an ultimate fate.

The public outputs should distinguish:

| Output | Interpretation |
| --- | --- |
| Product weight or amount | Expected availability in one declared pool and time scope; includes nonbinders |
| Product sampling distribution | Those weights normalized over the supported product universe |
| pMHC weight or amount | Availability of a peptide bound to a particular supplied MHC |
| Ligand sampling distribution | Normalized pMHC weights, with its own conditioning and denominator |
| MS response | Amount-dependent observation behavior under declared acquisition/search conditions |
| Observed elution prediction | Combined prediction at the scope and units of the actual measurement |

A softmax over products alone loses total amount. Keep nonnegative amounts or
unnormalized weights internally; normalize only for a declared sampling request
or conditional likelihood. A perturbation can reduce the entire pool even when
its normalized composition barely changes. Distinguish relative quantities,
absolute molecules, production flux and residence-weighted abundance.

Do not sum upstream availability and downstream bound abundance as if they were
disjoint states at the same time. A joint unbound/bound snapshot distribution
requires a common pool/time contract and, eventually, coupled occupancy dynamics.
The first model can expose these as separate scopes. A zero denominator produces
an empty/undefined distribution, not a uniform fallback. Unsupported binding
scoring does not erase a span from the processing distribution.

## Represent the products as a band of spans

Use a start-position by length array, with a mask for valid coordinates. For a
protein of length `L` and allowed lengths `l_min ... l_max`, the number of spans is

```text
N = sum_l max(0, L - l + 1)
```

This is linear in protein length for a fixed length range. A 500-residue protein
has 8,313 spans of lengths 4–20, or 19,220 of lengths 1–40. These are indexing
counts, not measured runtime estimates. Scoring every span against multiple MHC
molecules and class-II registers is more expensive than merely storing weights.

Use 4–20 as an optional bounded pilot, not a universal biological cutoff.
Fragment support, binding-model support and MS search support are three separate
masks. Longer intermediates and shorter degradation products need explicit
handling; material outside the chosen window cannot silently be assigned zero
probability. Class-II data include ligands longer than 20 residues
([Paul et al.](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2018.01795/full)).
Choose the real-data window from a frozen coverage audit; report truncation and
add longer sparse states where needed. An overflow category records missing
support but cannot substitute for the identity of those missing products.

Represent a span using boundary token vectors, pooled interior features, length,
source flanks and protein-boundary flags. Score the joint start/length choice;
independent start and end probabilities would impose an unnecessary restriction.
Keep actual peptide strings for the pMHC encoder. A weighted average of peptide
embeddings is not a physical peptide and does not preserve a nonlinear binding
prediction for every candidate.

Encode long source proteins in overlapping chunks, with explicit ownership of
spans and sufficient context at chunk boundaries. The current sequence encoder
returns token-level vectors, but the modular path has a 512-residue limit and
does not implement this long-protein contract. Preserve source coordinates;
never truncate proteins or double-count overlap. Cache identical peptide–MHC
computations independently of source occurrence, then sum genuine source
contributions. Duplicate export records are not extra material.

## Machinery must change sequence preferences

Reuse the current typed `Context`, `Component`, `Exposure` and
`MachineryProfile` records. Their meaning should remain stable across the
direct-assay model and this extension.

| Context | Conditioning and limits |
| --- | --- |
| APC species and cell type | Versioned reference priors and learned effects within supported data |
| Individual component states | Concurrent intact/reduced/lost/unknown states, with override and evidence |
| Measured activity or abundance | Numeric inputs with units and missingness, not inferred from a KO label |
| Cytokine state, dose and duration | Context only when those channels have training contrasts |
| Antigen delivery and compartment | Determine which supported stages apply |
| Route, compartment and MHC class | Supported class-specific processing/loading routes; separate from APC species |
| MHC molecules | Selection context and supported quantities; not a processing gene label |

A small nonlinear interaction between span features and the machinery vector
lets a condition change relative peptide preferences. A condition-only additive
offset would change total supply but disappear from the normalized distribution.
The model needs both a possible total-amount effect and sequence-dependent
effects; neither should be assumed to exist for an unseen condition.

Keep individual gene/component identities in the data even if sparse training
requires pooling some effects by functional group. Do not learn arbitrary
cell-line or study-ID embeddings as substitutes for machinery. Explicit sample
evidence overrides reference defaults; missing is not intact. Feature exposure
and matched control/perturbation data are required before serving an effect.

An isolated purified binding prediction must be invariant to these processing
inputs. Conversely, a component loss should not be implemented by blanking MHC
sequences or forcing all products to zero. A transport component such as TAP
belongs to a supported movement-between-pools stage; its action does not change
the peptide's amino-acid sequence. Its quantitative effect must come from data.

## The minimal statistical model

Use three learned components: conditional product supply, effective pMHC
selection/retention, and shared MS response. Retain the existing laboratory
digestion branch for non-MHC proteomics. Use the shared sequence encoder and
small fusion blocks rather than separate large networks for every named process.

One concrete factorization is:

```text
S(o,c)   = source_amount(protein(o), c) × exp(processing_score(span(o), c))
S(p,c)   = sum over genuine occurrences o of peptide p: S(o,c)
B(p,a,c) = S(p,c) × exp(retention_score(pMHC_representation(p,a), loading_context(c)))

mu_elution(p,r)
  = run_scale(r) × MS_response(p,r) × sum_a recovery(a,r) × B(p,a,c)

mu_bulk(p,r)
  = run_scale(r) × MS_response(p,r) × digest_supply(p,r)
```

Here `o` is a source occurrence, `p` a peptide/peptidoform, `a` an MHC molecule,
`c` a biological condition and `r` an acquisition. Compute positive sums in log
space where appropriate. If a source amount is unavailable, preserve an explicit
missing/prior contract; do not silently treat transcript levels as protein flux.
Use a common amount scale only where supported by actual standards or normalization.
Run scale and recovery need measured/reference contracts or explicitly fitted
nuisance parameters with a policy for new runs. They are not learned study-ID
embeddings. Within-run conditional fitting can remove an unknown total scale,
but then cannot establish absolute cross-run amount changes.

This is an **effective statistical factorization**. `S` is an availability
index until product data anchor it; the retention score is not automatically a
capture probability, a kinetic rate or a separate residence-time measurement.
Multiplication combines positive relative contributions, not independently
calibrated biochemical probabilities. Without component anchors, only their
combined observation prediction is identified. Capability reporting must say so.

An elution-only subset must therefore collapse unanchored supply/retention into
one effective availability function, rather than allocate two free scalar terms
that can cancel. It can serve observed-elution predictions, not a qualified
nonbinder pool. Expand to the factored form when product/retention anchors support
it. Freezing the pMHC encoder does not by itself identify a newly learned retention
function. This is a data-dependent model subset, with the missing product-pool
capability explicit, not a substitute for obtaining those measurements.

The factorization is not a particle-conserving simulation: a downstream snapshot
can depend on different residence times and source history. If predicting
time courses, depletion or competition becomes a requirement, replace the
effective retention modifier with a qualified flow/occupancy model. Do not bolt
an unconstrained extra score onto it and call the result physical kinetics.

For genuinely resolved molecule amounts, add contributions across MHCs. For an
unresolved multi-allelic measurement, marginalize with a declared mixture and
uncertain recovery/allele quantities. Do not duplicate the positive label across
all MHCs. The current model's log-mean-exp bag aggregation is statistical; moving
to quantity-weighted sums is a new contract, not a reinterpretation of that code.

## How affinity and stability participate

The existing pMHC encoder/fusion and class-II register handling produce a rich
representation. Direct IC50, KD, dissociation and stability data supervise their
own measured endpoints; the retention function consumes that representation.
This makes their learned sequence information available to the processing model
without defining `sigmoid(IC50)` as a biological capture probability.

Use qualified scalar assay predictions as additional inputs only in an explicit
comparison that tests whether they help beyond the representation. Preserve units
and assay distinctions. IC50 is not KD; thermal stability and cellular persistence
are not interchangeable; correlated affinity/stability outputs are not independent
probabilities to multiply together. Never feed an untrained assay head downstream.

First establish useful pMHC fits, then freeze them for a processing pilot.
Subsequent joint training should replay direct assay supervision and monitor
retained quality. If scalar assay predictions become downstream inputs, stop
elution gradients through those scalar readouts initially, so elution fitting
cannot silently redefine their measured meaning. Shared representation training
is a separate, declared choice. The same discipline applies to calibrated MS
anchors and auxiliary species/class predictors.

Class II retains the full candidate peptide, all supported core registers and
flanks. Aggregate one measured peptide once; attention weights are not physical
register occupancy without separate evidence. Other loading/editing/pMHC
interactions belong in the effective selection context until their observations
justify a separately supervised component. No unnamed spare head is required.

## MS detectability is conditional on amount and acquisition

The transferable question is: **given this peptide at this amount and these
acquisition/search conditions, how likely is identification, or what response
should we expect?** A single universal sequence-only detectability probability
cannot account for amount and acquisition. At a fixed reference amount and
instrument contract, a calibrated detection probability can be a useful output.
The detectability/abundance confounding is discussed directly by
[Li et al.](https://pubmed.ncbi.nlm.nih.gov/21067214/).

Train shared response features using non-MHC measurements and independent peptide
standards where available. Separate laboratory digest supply, sample recovery
and instrument response. Digest identity conditions the digest branch; it does
not become a detector shortcut. Similarly, MHC immunocapture recovery belongs
upstream of the detector. Normal proteomics is useful but does not by itself
isolate detectability or measure the endogenous unbound peptide pool.

Evaluate transfer across peptide lengths, terminal chemistry, supported
modifications and acquisition domains. Non-tryptic synthetic peptide resources
can help assess this transfer; the
[ProteomeTools extension](https://www.nature.com/articles/s41467-021-23713-9)
provides such spectra. A successful spectrum or fragment-intensity predictor
does not alone calibrate identification probability or recovery per molecule.

Choose an observation likelihood by actual measurement:

- Quantitative intensity: a declared log-scale noise/censoring model, with
  run scale and recovery accounted for. Raw intensity is not a molecule count.
- Identification within a verified search scope: an amount-dependent detection
  likelihood. Non-identification is an observation outcome, not absence of product.
- Counts: a count likelihood only for actual counts with a suitable sampling
  contract; do not relabel intensities or technical fractions as independent counts.

Missing measurements are masked. Below-limit measurements retain their censor
bound. Elution-only training can fit combined observability but cannot establish
the abundance of sub-ligand nonbinders. That output needs relevant product-pool
measurements and must remain unqualified until they exist.

## Differentiable discrete steps before RL

For a finite collection of spans, evaluate each candidate and sum its weighted
contribution. Gradients pass through the weights and candidate scores. This
marginalizes the discrete uncertainty; it does not replace amino acids with
continuous mixtures. Structured graphical inference can be embedded in neural
training in this way ([Structured Attention Networks](https://arxiv.org/abs/1702.00887)).

For mutually exclusive latent explanations, sum their weighted likelihoods
before taking the log. For coexisting molecules contributing to one measured
pool, sum their amounts before the observation link. These are different models;
neither justifies evaluating a nonlinear pMHC network on an averaged peptide
embedding or exchanging an expectation with a nonlinear loss.

For a bounded acyclic processing graph, sum over paths using dynamic programming
when the state captures all relevant history. The illustrative recurrence is
`weight(next) += weight(state) × transfer(state,next)`. The meaning of a transfer
must be declared: a probability for an exclusive fate and an expected yield for
branching cleavage are different quantities. A single softmax over every child
is not a model of producing several fragments. Check residue/material accounting
separately from molecule count.

Use transport, trimming, loading and removal as optional supported state types,
with class-specific routes. Add explicit transitions only when intermediate data
or an informative experimental design make the added parameters useful. Cycles,
history-dependent competition and continuous time are not covered by a simple
acyclic recurrence; they require a separate model and validation contract.
Sample paths afterward for diagnostics, with latent-history uncertainty visible.

| Framework | Recommendation |
| --- | --- |
| Weighted spans with exact sums | First implementation; simple gradients and a clear reference distribution |
| Sparse graph with dynamic programming | Next when explicit routes add supported structure; exact only under the declared factorization |
| Importance sampling or sequential Monte Carlo | Scale approximation when enumerating states or scoring all candidates is too expensive; retain proposal weights and effective sample size |
| REINFORCE with a learned baseline, or actor–critic | Reserve for unavoidable sampled discrete decisions; optimize the data objective and compare gradient variance with the exact reference |
| Gumbel/Concrete relaxation | Optional experiment for a specific bottleneck, with bias and temperature checks; not the default peptide representation |
| GFlowNet with trajectory balance | Later amortized sampling proportional to a validated nonnegative amount function; not evidence that the amount function is correct |
| PPO or reward maximization over binders | Poor initial fit to the task; high reward need not reproduce the product distribution |

The distinction between score-function and pathwise estimators is covered in
[Monte Carlo Gradient Estimation](https://www.jmlr.org/papers/v21/19-346.html).
Continuous relaxations introduce a different optimization approximation, with
bias relative to the discrete objective
([Concrete distribution](https://arxiv.org/abs/1611.00712)). There is no need to
introduce that approximation when the relevant sum is tractable.

If a learned sampler becomes worthwhile, prefer a distributional target over
maximizing mean affinity. GFlowNets target reward-proportional sampling
([foundations](https://www.jmlr.org/papers/v24/22-0364.html)); trajectory balance
is a candidate objective once terminal identity and multiple-path accounting
are specified. A high-abundance nonbinder must still receive sampling weight.
Freeze the target amount model during the first sampler comparison. An RL policy
that merely exploits errors in its own scorer is not a better biological model.

## Scaling and subsets

Start with exact enumeration on a small frozen human reference/cell-line cohort.
Batch proteins and spans, reuse source and MHC encodings, and avoid allocating a
dense proteome × span × allele × register tensor. Span storage is modest;
repeated neural pMHC evaluation can dominate cost. Benchmark those separately.

When subsampling, retain known inclusion/proposal probabilities and validate
weighted estimates against enumeration. Top-k pruning changes support and can
remove the nonbinders whose distribution we need. A sampled estimate inside a
log-likelihood can also be biased; an unbiased intensity estimate alone does not
prove unbiased likelihood gradients. Pre-register the approximation and convergence
checks rather than calling every sampled objective exact.

Use one canonical training path with losses masked by observation type:

| Data subset | Trainable path and claim |
| --- | --- |
| Purified pMHC assays | Existing pMHC representation and matching assay heads |
| Independent processing/product data | Span supply or the measured transition/pool only |
| Non-MHC digest observations | Digest supply plus observation response; isolated response requires anchors |
| Matched endogenous peptidomics | Relevant unbound/product pool plus its observation route |
| Elution observations | Combined supply, retention and MS path; decomposition needs anchors |
| Matched APC conditions | Supported conditional effects, retaining specimen and assay matching |
| Paired TCR observations | Receptor–pMHC scoring; no synthetic supervision of source abundance |

Every allocated branch must have a loss path and actual feature/update coverage.
Unmeasured stages are latent or absent, not supplied with fake labels. Preserve
subset compilation, freeze dependencies and exact-resume contracts. A model can
be useful for observed elution while remaining unsupported for separate product
amounts, transport probabilities or causal intervention effects.

## Fit with the current architecture

The following describes the current PR implementation, not a completed
processing simulator:

| Existing component | Reuse or required extension |
| --- | --- |
| `models/encoders.py` | Shared sequence features and token vectors; add audited long-protein chunking/span extraction |
| `models/modular.py` | Reuse pMHC fusion, explicit class-II registers, direct assay heads and optional TCR path |
| `data/modular_context.py` | Reuse species/cell/machinery resolution and override provenance |
| Cellular context fusion | Supplies contextual features today; add a product-supply path that does not require MHC inputs |
| Digest and shared MS blocks | Reuse feature/anchor structure; a calibrated amount-to-response likelihood is additional work |
| `data/modular_quantitative.py` | Preserve units, transforms and censoring; add explicit pool/flux/amount contracts where needed |
| Source admission and observation index | Preserve lineage and grouping; add complete source occurrences, pool states, candidate support and sample-level scope |
| Coverage and subset/resume machinery | Extend to span lengths, state types and transition/context features |

Currently, machinery context is constrained to cellular routes; there is no
MHC-independent processed-product endpoint. `cellular_quantity` measures pMHC
availability, not generic cleavage or transport. Add dedicated endpoint semantics
instead of relabeling those outcomes. The current MS component scores and
register attention must not be silently reinterpreted as calibrated rates,
amounts or occupancy probabilities. Keep the direct assay-only mode intact.

## Implementation and acceptance sequence

1. **Freeze data and estimands.** Audit actual source products, lengths, units,
   compartments, MHC sets, quantity standards, search denominators and matched
   conditions. Distinguish available Hitlist columns from upstream requirements.
   Choose candidate support and independent splits before fitting.
2. **Build the exact reference.** Add span indexing, masks, aggregation and
   machinery-conditioned supply with small enumerated fixtures. Verify counting,
   chunk ownership, source deduplication, finite gradients and exact normalization.
3. **Fit the minimal coupled model.** Use qualified pMHC/MS components or explicit
   frozen baselines. Compare against source-only, pMHC-only and current observed
   elution predictors. Do not claim an unbound distribution from elution alone.
4. **Test the proposed explanations.** Require sequence-dependent condition
   effects, retained assay/anchor quality and matched held-out contrasts. An
   ablation can establish predictive usefulness; a latent name cannot establish
   mechanism. Additional transitions must earn their parameters with data.
5. **Scale and sample.** Measure exact cost, then compare approximation methods
   at matched budgets. Require recovered mass, tail/nonbinder coverage and
   uncertainty as well as runtime. Add RL/GFlowNet only if it improves this tradeoff.

Before real fits, register numerical gates, seeds, baselines and artifacts under
the canonical experiment workflow. Report held-out likelihood, quantitative
error where supported, calibration, prediction spread and source/allele/length
strata. Preserve per-example predictions, sampled paths/weights and model hashes.
Split repeated measurements and homologous/source-related peptides honestly.

Required failure checks include an abundant nonbinder retaining its mass, zero
binding not zeroing supply, no detector change from an APC-only override, no
purified affinity change from machinery context, distinct all-pool and bound-only
normalization, and unchanged endpoints when equivalent paths are duplicated.
On known artificial distributions, compare gradients and sampling frequencies
with exact enumeration. On real data, require improvement over simple baselines
without a collapse to nearly constant outputs or loss of existing assay quality.

## Completed algorithm comparison — October 21, 2026

[Full report, source and reproduction bundle](https://github.com/pirl-unc/presto/blob/af74af65f6a65e4ead63988e4459c211e46ef3da/experiments/2026-10-21_1700_codex_discrete-distribution-comparison/README.md).

Both exact differentiable enumeration and discrete REINFORCE recover a known synthetic distribution over integer-symbol substrings. This is an experiment-local algorithm prototype, not biological peptide processing or real assay validation. The two-step policy samples an actual start and length and preserves repeated draws as a multiset. All methods share architecture, initialization and data; exact and sampled variants optimize the same reverse-KL objective.

| Method | Mean held-out total variation (0 is perfect) | Seeds |
| --- | ---: | --- |
| Uniform | 0.831593 | Unfitted |
| Exact differentiable | 0.010716 | 42/43/44 |
| REINFORCE, 8 samples | 0.013602 | 42/43/44 |
| REINFORCE, 64 samples | 0.011359 | 42/43/44 |
| Reward-only/no-entropy diagnostic | 0.253689 | 42 |

All three seeds of all matched methods pass the prospective validation screen. Reward-only optimization overconcentrates and fails the screen. Arbitrarily noncompatible products retain their target mass in the matched methods. The 64-sample estimator has approximately 8.5 times lower summed gradient variance than eight samples in the prespecified fixed-batch probe; both still compute dense policy logits, so logical reward-query savings do not establish a scalable runtime advantage.

Six focused tests, independent teacher/metric/sample reconstruction and fresh-process replay of all ten selected checkpoints pass. The report preserves full train/validation/test distributions, sampled multisets, source/seed/environment receipts and a verified compressed raw bundle. No synthetic checkpoint is promoted into Presto.

Decision: retain exact enumeration as the finite-support reference and discrete distribution-matching REINFORCE as a working comparator. The known-oracle test does not establish a real processing distribution, species or machinery effects, latent-path identifiability, learned pMHC/MS factors, or resolution of IC50 underfitting. Those parts of this issue remain open.

## Completed GRPO-ingredient factorial and hybrid comparison — October 22, 2026

[Full 69-fit report and verified reproduction bundle](https://github.com/pirl-unc/presto/blob/59bf87177ab2f24fb674d270f2326aad06dd3946/experiments/2026-10-22_1120_codex_grpo-ingredient-factorial/README.md).

The full clipping × group-standardization × reuse × reference-KL factorial has run, with three seeds, exact and original REINFORCE controls, two hybrid budgets, and separate query-matched reuse conditions. All 69 fits completed and passed the basic validation recovery screen on the unchanged integer-symbol task. This is not a biological concentration result.

| Condition | Mean test TV to original target | Logical fitting target terms |
| --- | ---: | ---: |
| Exact enumeration | 0.010716 | 1,612,800 |
| Original REINFORCE-8 | 0.013602 | 184,320 |
| Group-standardized, fresh samples | 0.016237 | 184,320 |
| Reuse 4, equal optimizer budget | 0.025839 | 46,080 |
| Reuse 4 plus clipping, equal optimizer budget | 0.023572 | 46,080 |
| Reuse 4, equal target-query budget | 0.019524 | 184,320 |
| Hybrid 2 starts, enumerate lengths | 0.017576 | 174,864 |
| Hybrid 8 starts, enumerate lengths | 0.012454 | 698,104 |

Clipping is inactive for fresh-policy first updates, as verified by paired predictions. It only modestly helps one reuse comparison. Group standardization worsens mean recovery here. Query-matched reuse spends fourfold optimizer work and still loses to fresh REINFORCE. Hybrid eight-start enumeration helps at greater scoring cost; the roughly query-matched two-start hybrid does not. The beta=0.1 reference penalty changes the optimum: exact fitting has TV 0.011521 to its regularized target but 0.041213 to the original target.

The implementation uses recomputed density costs and joint span importance ratios, so these are GRPO-style ingredient ablations, not canonical fixed-reward/token-wise GRPO. Both methods still compute dense policy scores. Seventeen local checks, independent reconstruction of 1,035 split/context metric sets, all 69 selected-checkpoint replays, nine initial-reference replays and 18 bitwise matches to previous exact/RL predictions pass. All four exact-head CI jobs passed.

Decision: retain exact enumeration and simple REINFORCE as references for the next finite partial-observation and whole-multiset study. Its prospective contract separates unbiased distribution matching from naive histogram sampling variance and tests unidentifiable unseen mass. Real biological source qualification, precursor-normalized availability and calibrated concentration remain active work; this issue remains open.


## Completed finite-observation and whole-multiset comparison — October 22, 2026

The [60-fit report and complete reproducibility bundle](https://github.com/pirl-unc/presto/blob/ed340662154c116317b77bf005aa28154f0c3e61/experiments/2026-10-22_1317_codex_partial-observation-multiset/README.md)
are published on draft PR #131. This closes the requested synthetic partial-
observation/multiset stage after the 69-fit optimizer/hybrid stage above.

Same integer-symbol model and parent-disjoint splits, but training receives only
known observation weights and finite counts. Cross positive/blind observation
weights × 32/256 nested counts × exact CE, exact L2, corrected multiset L2 (16/64
draws, four independent groups), and naive histogram MSE × three seeds. Oracle
latent probabilities are excluded from training and checkpoint selection.

Mean test TV at 256 observed counts (lower is better):

| Method | Positive observed / latent TV | Blind observed / latent TV |
|---|---:|---:|
| Exact likelihood | 0.02472 / 0.02679 | 0.02335 / 0.04419 |
| Exact L2 | 0.03175 / 0.03324 | 0.02990 / 0.05116 |
| Corrected multiset 16 | 0.05745 / 0.05978 | 0.05012 / 0.07811 |
| Corrected multiset 64 | 0.04855 / 0.05035 | 0.04131 / 0.06072 |
| Naive histogram MSE | 0.06634 / 0.07324 | 0.05532 / 0.08041 |

Exact CE has the lowest mean observed/latent TV in all four data contracts.
Corrected multiset objectives learn, but trail objective-matched exact L2.
Exhaustive checks verify that naive histogram MSE adds a sampling-variance term,
whereas corrected multiset costs/gradients have the intended L2 expectation.
More generated samples usually help, but cannot supply missing observations.

All 60 fits completed. Eight pre-fit tests, independent reconstruction of 576
teacher rows, 1,152 observed-count rows and 924 metric sets, all 60 training-row
checkpoint replays, and all 368 raw archive members pass. All six parameter
tensors update. Complete held-out predictions, counts, metrics, contexts,
checkpoints and logs are retained. Family runtime 72.504 seconds on one CPU
thread; no GPU/Modal. All methods still enumerate dense span logits; sampled
variants add draws, so this does not establish large-support sampling speedups.

The blind condition hides mean test latent mass 0.73125. Recovery under this
shared parametric teacher does not establish identifiability: two distinct latent
distributions with different invisible mass can have exactly the same observed
distribution. Conditional counts also cannot establish total concentration.

Decision: retain exact likelihood/differentiation when the observation likelihood
is tractable, and the corrected multiset estimator as a discrete reference for
collection-level or nondifferentiable scoring. No biological model promotion.
The next stage is source admission of endogenous processing-product measurements,
paired precursor proteomics and quantitative APC immunopeptidomics. Actual source
columns, pairing, units, capture scope and missingness must be audited before a
biological fit. This issue and the biological availability goal remain open.


## Biological source admission checkpoint — October 22, 2026

[The PXD057588 initial source audit](https://github.com/pirl-unc/presto/blob/51b895eb3925de87df7710c7bd6c78d131c25697/experiments/2026-10-22_1410_codex_availability-source-audit/README.md) is published on draft PR #131. Seventeen pinned assets, six quantitative workbooks and the community SDRF were inspected; independent verification covers 309,126 quantity cells and 3,240 SDRF cells. This is source qualification progress, not biological model fitting.

Important distinctions are now concrete acceptance requirements in [Hitlist #736](https://github.com/pirl-unc/hitlist/issues/736) and [#737](https://github.com/pirl-unc/hitlist/issues/737): measured macrophage versus MHC-loss tumor partner; digest-peptide versus protein quantity; eight-of-nine differing cross-assay TMT mappings; partial SDRF coverage (six of fifty raw files); conflicting replicate/state annotations; independent spreadsheet panels; selection/normalization/calibration scope. These must be preserved in the common catalog and frozen downstream bundles, not a parallel Presto source registry.

Normalized source curation, exact specimen joins, endpoint admission and real-data fitting remain pending. MHC-eluted signals do not directly label the whole intracellular peptide pool. The Wolf-Levy processed search archive download remained incomplete and was excluded. The availability/concentration objective and this design issue remain open.


### Biological quantity qualification checkpoint — 2026-10-22

Published [PXD057588 quantity reconciliation](https://github.com/pirl-unc/presto/blob/30cc5368bfa58b936339c3e5600309625831d851/experiments/2026-10-22_1601_codex_pxd057588-quantity-reconciliation/README.md) at Presto 30cc536.
All200,724 primary fold-change cells reconcile (maximum error 1.63e-14); independent
scalar replay/source coordinates and parent-support counts pass. The deliberately
opposite normalization disagrees on every cell. Exact source controls, operation
order and normalization universe are required in the shared [Hitlist contract](https://github.com/pirl-unc/presto/blob/30cc5368bfa58b936339c3e5600309625831d851/docs/study_curation_contract.md),
with upstream #736/#737 updated. Only 110 shared parent labels have both multiple
observed ligands and multiple digest peptides (258 ligand rows), constraining a
within-parent distribution cohort. No biological fit or concentration claim.

The separate Hitlist worktree is implementing canonical context PR 3; this thread
stopped its isolated competing prototype and focused on source qualification.
Normalized PXD ingestion, sample/reference joins and endpoint admission remain
pending. The previously partial Wolf-Levy PXD009092 SEARCH.7z subsequently completed:
51,736,986 bytes, deposited checksum and full archive decode pass. Its immutable
receipt/recovery recipe are in the initial source-audit directory; the original
17-asset audit contract is unchanged. Transfer integrity alone does not qualify
those processed tables for training. The availability goal remains active.


### Biological contrast source checkpoint — 2026-10-22

[Published source report](https://github.com/pirl-unc/presto/blob/ef4b4c1a51e74add1557b8aa5fcc8d95a84309c5/experiments/2026-10-22_1630_codex_mapp-source-admission/README.md), Presto ef4b4c1. PXD009092 supplies an actual
cell-line UT/cytokine candidate: 2,272 structurally supported peptides across 572
protein groups, including 271 groups with multiple peptides. Independent parsing
verifies 149,959 source evidence rows, 160,095 quantity comparisons and all 10,673
row-level support decisions. No model was fitted and no biological row admitted.

The audit identifies 9,276 positive-evidence entries exported as zero, discordant
peptide-FDR claims, and 36 group records without IDs; all are explicit shared
Hitlist acceptance cases in pirl-unc/hitlist#737. Captured-fragment protein-group
LFQ must not masquerade as precursor abundance. The candidate independent tryptic
archive PXD009360 is fully checksum/CRC verified, but its nine numeric experiment
labels lack treatment/specimen mappings. Source context and exact versus surrogate
pairing must be resolved before precursor adjustment.

The next fit-specific contract will compare the fixed exact and fresh REINFORCE
references on the same observed relative-signal endpoint and protein-group holdouts,
with constant/length-composition baselines and complete held-out dumps. Resolve
source filtering, confidence and homogeneous metadata admission before launch.
This concrete biological support advances beyond the synthetic experiments; it
does not establish free-pool concentration, APC generalization or completion of
the availability goal. Prior head 30cc536 passed all four CI checks; ef4b4c1 has
local strict docs, audit lint and independent source verification completed.


### Local training feasibility — October22,2026

The user has ruled out Modal as the execution assumption. A completed local
comparison and full-cohort one-epoch IC50 fit are recorded in
[the report](https://github.com/pirl-unc/presto/blob/e80fd42/experiments/2026-10-22_1745_codex_local-ic50-feasibility/README.md).
On the M1 Pro, canonical audited updates took median0.427sCPU versus0.487sMPS
(8measured after4warmup; preliminary short comparison). One complete CPU epoch
on21,636admitted exact observations, including evaluation, took169.97s and
2.59GiB peak processRSS. All32allocated tensors updated; testRMSE1.1613 versus
ridge1.2291, Spearman0.5117 versus0.4306. Predictions are nonconstant.
Independent82,092-record verification, coefficient refit and fresh-checkpoint
training replay passed.35training/resume tests pass after a constant-baseline
Pearson reporting correction.

This supports continuing locally, not treating this one-seed/one-epoch result as
a completed model. Test binderF1 is essentially tied with ridge, classII binder
recall is0.0882 and0/203test bounds are satisfied. These are purified binding
measurements, not availability labels. The biological availability objective,
shared source-curation work and MAPP publisher reconciliation remain open; no
intracellular concentration claim or all-module qualification is made.


### MAPP publisher reconciliation completed — October 22, 2026

[Source report](https://github.com/pirl-unc/presto/blob/4d938e0/experiments/2026-10-22_1736_codex_mapp-publisher-reconciliation/README.md), commit4d938e0. All160,095deposited cells reproduce under the fixed run-intensity-versus-two-times-median-beads rule. Publisher UT/TNF+IFN memberships and4,983/5,311nonimputed log medians match exactly. Independent checks cover1,440,855rule decisions and229,160publisher/source comparisons.

Imputed E/F columns have a condition-label conflict: each matches only two rows under its own label, but thousands under CHX/UT instead. They remain excluded from measured training inputs. Leading-razor assignments retain alternative origins;559/604publisher rows have multiple source group IDs. The selected131-gene Data3 summary supplies no raw proteome specimen/run map.

The new source-generation work is assigned to [Hitlist #746](https://github.com/pirl-unc/hitlist/issues/746), which receives the exact arithmetic, missingness and ambiguity fixtures. Relative observed-fragment composition can be the first real local comparison after shared import/reference/endpoint qualification. This turn fitted no biological model and makes no latent concentration, complete peptide-pool or professional-APC generalization claim. The full goal remains open.


## Verified MAPP parent-reference mapping — October 22, 2026

The [reference/position audit](https://github.com/pirl-unc/presto/blob/6cd7edc/experiments/2026-10-22_1845_codex_mapp-parent-reference/README.md) is complete. This follows the quantity and publisher reconciliation, and changes the biological modeling/split contract: parent ambiguity must be retained explicitly.

- Two immutable reviewed-human reference contracts remain distinct from the unavailable original `Human_UP_2016_02.fasta`: a local 2023 canonical DAT with recorded secondary aliases and an official release 2026_03 isoform-inclusive FASTA with exact IDs. No suffix stripping, I/L equivalence, unrecorded alias transfer, or source-coordinate repair.
- In the fixed 2,272-peptide structural contrast subset, 2,269 match the current reference. Of those, 1,013 match one distinct full sequence and 1,256 match multiple full sequences. Only 1,676 agree with the resolved source-leading coordinates and flanks; 26 have resolved coordinate conflicts. Three remain unmapped. These are mapping facts, not source admission or concentration labels.
- Preserve all source/accession/parent/span/sequence-hash alternatives at their own grain. A peptide measurement must not be multiplied into independent labels by alternative-parent joins. Marginalization requires an explicit mixture model; a leading-razor assignment does not supply measured precursor-mixture weights. A stricter unambiguous diagnostic must be declared prospectively and report its coverage cost.
- The complete all-source exact-parent-sharing graph has 542 components intersecting the contrast subset, with 175 subset peptides in the largest. These constrain future train/validation/test grouping but do not establish homology or specimen independence. No predictive split or fit was performed here.
- Independent Biopython parsing and a separate complete-reference character automaton reproduce all 37,721 position alternatives across both references, all 64,538 source-accession rows, and all subset/component summaries. Producer 11.537 s; verifier 17.734 s, local CPU. The frozen report includes full matched parent sequences, original fields, missing/negative candidates and exact reproduction artifacts.

The shared external-source mechanism and PXD057588 are now merged in [Hitlist PR #749](https://github.com/pirl-unc/hitlist/pull/749). MAPP remains a contract candidate. [Hitlist #752](https://github.com/pirl-unc/hitlist/issues/752) now specifies its actual source-generation adapter in that same package, with arithmetic/reference conservation, context/confidence/observability distinctions and task-specific admission. This replaces neither the broader architecture nor the availability objective: captured proteasome fragments are not free intracellular concentration; shorter unobserved products, APC machinery effects, independent precursor abundance and downstream presentation/detectability remain in scope.


## Shared PXD057588 mouse-reference acceptance evidence — October 22, 2026

The [shared-generation admission report](https://github.com/pirl-unc/presto/blob/24f1d2b/experiments/2026-10-22_1914_codex_pxd057588-shared-sequence-admission/README.md) consumes the actual Hitlist 1.82.22 generation (manifest daff7930…), not a separate workbook-to-training bypass. Native verification and an independent full 87-file inventory/hash check pass.

- A pinned reviewed-mouse isoform-inclusive reference (UniProt 2026_03, SHA256 41bc0fa6…) gives exact matches for 35,771/35,772 source rows. 35,747 match their reported parent ID; 24 source IDs are missing and one resolves to a sequence without the peptide. Retain all original records and alternatives.
- 11,725 source rows match multiple full parent sequences. Across the 30,587 distinct peptides, an independent complete scan reproduces 52,277 positions. Source IDs, full sequences, reference generations, occurrences and biological origin have different meanings.
- Tables 9/11 have 661 shared source-parent IDs verified by peptide occurrence, versus 663 literal shared labels. 110 verified groups have at least two ligand and two digest rows, covering 258 ligand/1,919 digest rows. Additional exact alternatives yield 1,134 shared full parent sequences, with 236 multi-peptide groups; these add candidate ambiguity, not independently measured proteins. The report supplies exact support memberships.
- Carry mapping evidence through the same typed package/query path, preserving one measurement identity through alternative-parent joins. The 5,374 complete peptide-parent components (largest 862 distinct sequences) are grouping constraints, not homology or specimen independence.
- Mouse species exists in the original PRIDE project asset, but the current external typed source-assertion predicates do not materialize species. Explicitly expose source-backed species roles and unknown/unsupported context in the recipe; do not default these panels to human. Keep measured macrophage versus tumor partner separate, and do not put Table 10 tumor MHC-I loss on the macrophage.
- Preserve `assay_channel_unresolved_specimen`: a verified common parent does not establish matched aliquots/specimens across Tables 9/11. Digest-peptide intensity is not already a precursor-protein estimate. A later recipe must declare aggregation, ambiguity, normalization and exact versus condition-surrogate pairing, with per-row admission reasons and coverage. Keep complete-case selection and source conflicts.

Independent verification covers 71,544 original sequence/parent worksheet cells, all 35,772 source mapping rows, all 26,449 parent-support records and all seven table/global summaries. No predictive split or fit occurred. The next biological test should compare sequence/context predictions of observed condition-dependent signal with a parent/context baseline, optionally testing an explicitly labeled condition-surrogate precursor adjustment on verified support. Exact and REINFORCE conditions must share the observable objective and holdouts. This does not establish intracellular concentration or remove the broader availability goal.


### Variable-length availability policy and measured computation (October 22, 2026)

Implemented exact enumeration and lazy start/length sampling in `models/peptide_availability.py`; [registered report and complete reproducibility bundle](https://github.com/pirl-unc/presto/blob/80d5f9c/experiments/2026-10-22_1938_codex_variable-span-computation/README.md), draft PR #131. This is a computational prerequisite, with **zero optimizer updates and no biological fit**.

Eight unmodified full proteins from the verified human/mouse reference audits, four predeclared length bands per species, supports 1–40 and 1–53, fixed synthetic context and energy. The 5,536-parameter policy covers short/nonbinding products. It scores all starts, then conditional lengths only at distinct sampled starts; repeated draws remain in the output. CPU only, 4 warmups + 12 measurements per condition, 1,280 calls total.

For the 2,240-residue human protein, exact expectation/backward takes 31.37 ms; dense 64-draw REINFORCE takes 17.31 ms; lazy 64-draw REINFORCE takes 3.36 ms (5.15× faster than dense, 35.65× fewer conditional-network rows). Across 16 cases, dense/lazy runtime ratios are 2.06–8.79× for 8 draws and 1.33–6.91× for 64. All methods retain the O(L) start calculation. These timings do not establish convergence speed.

All 512 seed-matched dense/lazy sample pairs agree; selected log probabilities match exactly and maximum parameter-gradient difference is 1.77e-8. Independent NumPy features/MLPs/analytic backward reproduce all 678,190 probabilities and 16 exact gradient vectors. Fourteen tests, Ruff and strict docs pass. Averaging 128 REINFORCE estimates at 64 draws leaves about 10% relative gradient error on both longest-protein probes: equivalence of dense/lazy computation does not eliminate sampling noise.

**Decision:** keep exact enumeration as the reference and use lazy conditional scoring for sampled experiments. The primitive is not yet integrated into the canonical mixed-assay trainer. Shared Hitlist [recipe #739](https://github.com/pirl-unc/hitlist/issues/739) and [MAPP adapter #752](https://github.com/pirl-unc/hitlist/issues/752), biological holdouts and parent ambiguity, precursor yield, clearance, APC machinery, MS response and concentration qualification remain required; this does not complete the broader availability goal.


### Actual shared MAPP quantity import — October 22, 2026

[Hitlist draft PR758](https://github.com/pirl-unc/hitlist/pull/758) implements the quantity-ingestion portion; [registered source/verification report](https://github.com/pirl-unc/presto/blob/396f771/experiments/2026-10-22_2212_codex_shared-mapp-import/README.md). The first real two-source external generation preserves 10,673 MAPP peptides, 149,959 evidence rows, 18 runs, 362,882 quantity representations and 540,034 derivation edges. Independent original TSV/workbook-XML replay verifies all 111 package files, 162,767 raw records, all typed quantities/edges and 36,207 workbook cells. All 160,095 filtered values reproduce within the frozen source-rounding tolerance (maximum relative error8.16e-5), and all measured publisher medians reproduce.

Missing confidence is explicit: 74,573 source PEP values are literal NaN. The 9,276 filtered zeros with positive evidence remain selection outcomes, and imputed publisher-header conflicts retain no assigned condition. Full normal tests pass3,224/10warnings;16focused tests and format/lint pass. Producer43.29s; independent replay9.46s, local CPU.

This is a draft external-source generation, not a rebuilt query/training export or biological fit. Reference-scoped occurrence integration and endpoint admission remain unfinished; keep this issue and the full availability objective open. [Hitlist757](https://github.com/pirl-unc/hitlist/issues/757) now separately tracks recovery of PXD009360 run-to-condition/specimen metadata and a justified independent precursor-abundance recipe. Do not infer this denominator from fragment LFQ or the selected publisher protein panel.
