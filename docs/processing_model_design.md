# Processing distributions and MS observation design

Proposed extension to [issue 130](https://github.com/pirl-unc/presto/issues/130),
[issue 132](https://github.com/pirl-unc/presto/issues/132) and the
[modular model](modular_model.md) in draft PR 131, October 21, 2026.

Biological source admission follows the shared
[study and sample curation contract](study_curation_contract.md), implemented
through Hitlist's unified catalog. The complete new-source metadata path remains
pending qualification; per-study experiment notes do not replace it.

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
