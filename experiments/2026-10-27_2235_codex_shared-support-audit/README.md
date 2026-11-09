# Shared-model failure, weight and source audit

**Completed audit; prospective predictive follow-up is separate.**
Date: 2026-10-27. Agent/model: Codex / GPT-6. Baseline commit:
`2f5705184f6997d50883639f5d38f6c133b81995`. Implemented common-core fixes:
`0c4c984b` (configurable dimensions, same-coordinate update evidence, fixed-step
validation with exact recovery, and strict literal quantitative-assay admission).
[Follow-up comparison](../2026-10-27_2255_codex_shared-multi-assay-scale/README.md).

## Scope and reproducibility

No new predictive fits or heldout model inference in this audit. Six saved v5
checkpoints from the [prior comparison](../2026-10-27_2100_codex_transfer-improvements/README.md)
were inspected: joint, I-only and II-only, seeds 42/43. Initial, selected and final
weights remain in `artifacts/2026-10-27_2100_codex_transfer-improvements/runs/`.
The corresponding canonical validation/test prediction dumps remain under that
family's `results/{condition}/`; this audit references their exact paths and hashes
in each `*-failures.json`, rather than duplicating identical predictions.
Training-only gradient probes use the same 64 saved training rows per checkpoint,
in evaluation mode, without any optimizer step. They are diagnostic gradients,
not a reconstruction of historical element-level gradient exposure.

Run `reproduce/launch.sh` for the original weight/saved-prediction analysis.
`reproduce/launch.json` also records source preparation, independent verification
and failure-summary commands. Source snapshots are in `reproduce/source/`.
Local CPU, Python 3.12.6, PyTorch 2.14.1, NumPy 2.5.4; numerical-library thread
counts1. No requested/observed GPU for this audit. Original fits used H100! as
recorded in their family; this analysis must not be presented as a new GPU study.
No pretraining or synthetic labels. Existing IC50 regression uses standardized
log10(nM), literal original bounds and one-sided squared loss; source-only/joint
fits and sampling budgets are fully specified in the prior family.

## What actually trained

All six selected models have515,457 parameters. Tensor-level evidence reported
no untouched trainable tensors. Comparison with initialization found exactly128
unchanged coordinates: the fixed padding embedding row. All20 residue embedding
rows and all386 pMHC-fusion input columns have recorded input-column updates.
This does **not** establish element-level historical gradient exposure or
identifiability. Old checkpoints never stored those element masks.

There are384 structurally redundant attention key-bias coordinates across the two
encoder attention blocks and shared interaction. Adding the same bias to all keys
adds a query-dependent constant across attention logits, which cancels in softmax.
Their probe gradients were around1e-9, versus around1e-2 for query/value slices;
floating-point movement had nevertheless marked them as updated. A controlled
float64 perturbation test confirms evaluation invariance. Keep them for checkpoint
compatibility and report the redundancy explicitly; they are not evidence of a
missed biological training path or the cause of the observed prediction failures.

The two human-species inputs are constant throughout this cohort. Their weights
receive gradients but their effects cannot be separated from each other and the
fusion intercept using these data. Missing-species patterns and nonhuman transfer
are not validated. Weight movement alone is insufficient evidence of an identified
input effect. Unallocated cellular, MS, digestion and TCR modules contain no random
weights in these IC50-only checkpoints; their outputs are absent, not trained.

The common trainer previously counted a tensor as trained if it had a nonzero
gradient somewhere and movement somewhere, potentially on different coordinates.
It now requires both on the same coordinate, stores cumulative Boolean evidence,
and preserves unknown historical evidence on old checkpoints. New reports separate
fixed padding and structural key-bias redundancy. No coverage gate was loosened.

## Predictive failure modes

These are posthoc diagnostics on previously inspected development holdouts, not
independent qualification or causal identification. [Figure](results/failure-audit.png)
and [complete summaries](results/failure-summary.json) preserve the evidence.

| Joint v5 test diagnostic | MHC-I, seeds42/43 | MHC-II, seeds42/43 |
| --- | --- | --- |
| Pooled exact Spearman |0.519 /0.530|0.526 /0.482|
| Macro within-groove Spearman, groups with >=20 exact values |0.332 /0.311|0.528 /0.503|
| Prediction SD / target SD |0.631 /0.654|0.588 /0.662|
| Original censor bounds satisfied |3/87 and1/87|13/116 and12/116|

Scores compress toward the middle. For class-I subnanomolar examples (61 exact
values), mean log10 error is about+2.09/+1.99; for class-I values in the100,000nM
bin (105 exact values), it is about-1.61/-1.69. Class-II10–100nM examples are
likewise overpredicted by about1.15/1.11log10 units. These are statistical failures,
not proof of a particular architectural cause. Bounds were excluded from the old
training set, making original bound supervision a direct next comparison.

Some class-I grooves rank poorly despite positive pooled correlation: B*15:01
(n25) has negative Spearman in both seeds; other small groups vary substantially
by seed. Narrow target-bin correlations are range restricted and should not be
interpreted as general discrimination measures. `failure-groups.csv` contains
length, groove pair, qualifier, target-range and source-study groups. The additional
`novelty-register-strata.csv` records exact9mer overlap, seen groove/study status
and candidate counts. Exact-peptide separation is not protein-family or study
separation. Read group counts alongside every correlation.

## Source expansion, independently checked

Pinned Hitlistv4 exports from `2026-10-16T07-20-43Z`:
- `binding.parquet`: SHA256 `2b3f6ea2dd647e7fc54d214eef420feaebc4e81fab97ffc0565a2de1f8695ba0`.
- `other_assays.parquet`: SHA256 `c40f5227ded408830885ecba8336dbdd701172df9573b0d32e5e8a5f8fc947c2`.
- mhcseqs catalog: SHA256 `ddc3493c64f4328de174a6c2cda1f007ec2f93a1ef649add25d2ef9c1720add1`.
- Historical membership: SHA256 `8959e60c0d9b8049aec25095b9c354d456b48f97ba8fb31d7b786e2861113e99`.

Scanned840,805 raw rows. Retained the27,364 frozen IC50 rows byte-for-byte and
admitted140,846 additional outcomes: total168,210 observations,31,011 distinct
peptides and147 verified groove pairs. New outcomes include1,919 IC50 bounds,
18,817 directKD,388 EC50,53,947 KD(~IC50),56,190 KD(~EC50),9,575 half-life and10
off-rate measurements. These remain seven distinct literal outputs; no proxy is
relabeled as direct affinity. Only source-explicit purified-MHC assays, canonical
unmodified peptides, positive raw physical values, agreeing numeric text/qualifiers,
original assay identity and complete named human partners are admitted.

55,371 rows were deferred for missing explicit partners,1,832 for unresolved
catalog names; modified/unsupported/population/method exclusions are fully recorded
in `results/data-admission.json` and the compressed source-disposition ledger.
No DRA identity or other partner was guessed. Complete source preparation took
719.2seconds locally. No duplicate/conflicting measured outcomes remained among
new candidates;2,172 independent same-input disagreement groups were retained.
Their differing assay conditions are not matched, so these are not an irreducible
noise estimate. Training-only disagreement summaries are in the independent audit.

Across endpoints, connected peptide/original-assay components inherit frozen
historical assignments; conflicts would be deferred. Independent verification
checked **all168,210 raw-record fingerprints, original identities, targets/units,
qualifiers, source-named catalog groove pairs and split assignments**. It found
zero peptide leaks, zero cross-endpoint assay leaks and zero duplicate original
outcomes. Initial verifier attempts exposed two namespace conventions, not data
changes: the catalog uses `HomoSapiens-` aliases for `HLA-`; historical source
ordinals are one-based and new ordinals zero-based. Both are explicit in the final
verification. Frozen old records were not rewritten.

Cohort: `data/modular_cohorts/2026-10-27-purified-multi-assay/` in the durable
shared-training checkout. Source raw caches and exact local paths are frozen in
`reproduce/source/prepare_data.py`. All generated source files and their hashes
are listed in the cohort manifest. Large preparation SQLite/JSONL ledgers remain
under that cohort's `prepared/` directory. No scientific dataset was synthesized.

## Decision and software validation

Carry forward the existing shared-v5 architecture. Add configurable shared capacity,
honest coordinate evidence and step-based evaluation to the same trainer. Compare
original bounds, additional measured endpoints and increased capacity separately.
Six outputs have heldout examples; off-rate has ten training-only rows and is
excluded from the predictive follow-up. Inputs remain peptide and verified grooves;
no new cell/TCR machinery context is asserted. Previously unavailable TCRvdb and
unverified biological context remain separate data gaps.

296 focused common-model, inference, admission, continuation and resume tests pass
on commit0c4c984b, including every reachable route at nondefault width, historical
checkpoint behavior, disjoint-coordinate rejection and interruption during pending
interval validation. Production-file lint and diff checks pass. This establishes
software behavior, not improved predictive accuracy; that outcome belongs to the
separate eight-fit H100 comparison.

The preparation runtime is preserved in the follow-up family's
`reproduce/source/runtimes/shared/presto/` at commit0c4c984b. Use that directory's
parent on `PYTHONPATH` when rerunning canonical `code/prepare_data.py`; its original
preparation cohort directory must be absent. The initial analysis used baseline2f570518;
source preparation used the new strict adapter now committed at0c4c984b. Phase-specific
commands, source snapshots and the interpreter-field correction are explicit in
`reproduce/launch.json`.

A subsequent provenance-only fix adds the shared sampler module to generic resume
fingerprints. Its dedicated changed-source test passes before any resumed update.
The running comparison remains pinned to0c4c984b; its full Modal runtime bundle
already independently hashes/verifies the sampler. No running fit was modified.

## Register identifiability addendum

Posthoc analysis of the already saved predictions found diffuse class-II register
weights. For the two joint-training seeds, mean normalized entropy on the958
multi-register test observations was0.990/0.989 (uniform=1); median maximum weight
was0.182/0.185, only1.28/1.30times the uniform weight. No example assigned a register
more than half the total weight. The top start agreed between seeds on63.3% of
examples, compared with a14.2% uniform-random agreement reference. Class-II-only
fits were also diffuse (entropy0.984/0.983). This is weak statistical preference,
not established physical core identification: no measured register truth was used.
It does not establish that the preferred registers are wrong, nor does numerical
movement of all relevant weights establish learned register discrimination.

Reproduce with the recorded Python executable and
`analysis/register_diagnostics.py`; `results/register-diagnostics.json` retains
per-example diagnostics and both validation/test summaries. This consumes saved
outputs only, performs no new inference and does not alter the next family's
prespecified selection gates.

## Final software integration

`results/integration-verification.json` records sourceceebfb80 and the final
installable distributions. The broad suite passed2,767unique cases across its
initial run and targeted reruns, with four skips. Three initial failures were
missing historical files in the sparse checkout; restoring those tracked files
resolved all three. The initial collection failure and subsequent logs are retained.
All300focused shared-model tests pass. Full pinned Ruff lint/format checks, strict
MkDocs, distribution metadata checks and isolated installed-wheel execution pass.
All209Python files in the final wheel match the working source byte-for-byte.
The smoke now covers configurable shared capacity, both candidate rules and
untrained-output refusal, in addition to historical resources and loaders.

New training also makes checkpoint priority explicit through
`training.selection_endpoint`; null means the observation-weighted assay loss,
and the IC50 recipe explicitly selects IC50. The predictive comparison retains
its frozen0c4c984b runtime and original selection contract. No running fit or
selection gate changed, and the shared virtual environment was not modified.
