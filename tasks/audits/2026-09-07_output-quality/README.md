# Output supervision and prediction-quality audit

Date: 2026-09-07. Auditor: Codex. Code inspected:
`1535b5a208b2ab59b70b83d850b1f0a7e1959537` (PR #47, still open at inspection).
All four CI checks on that head now pass. This audit covers existing gaps as
well as PR #47's bounded repairs; it is not limited to regressions introduced
by that PR.

## Conclusion

The implementation can execute a differentiable training path, but “all outputs
are supervised and make coherent predictions” is not established. Six actionable
work items follow from verified code gaps and the existing experiment record:

1. Make coverage measurable per output/column, source and split, including MIL.
2. Preserve binding assay descriptors in the merged TSV adapter.
3. Train observed T-cell assay columns on pathway-MIL observations.
4. Make held-out artifacts cover the same observations/outputs as training.
5. Correct average precision for tied prediction scores.
6. Establish a fresh real-data baseline with ground-truth-linked examples.

These are scoped follow-ups to [#46](https://github.com/pirl-unc/presto/issues/46).
The wider biological role/schema work remains there. No new optimization run,
GPU job, performance benchmark, or predictive-quality evaluation was conducted
for this audit. Code diagnostics use artificial fixtures and an untrained model;
historical corpus measurements retain their original contract and date.

## Evidence and reproduction

- [diagnose.py](diagnose.py): deterministic assertions exposing the gaps; **zero
  optimizer steps**. These assertions intentionally describe the audited
  pre-repair revision. Run with production imports from `1535b5a`, copying this
  audit bundle into a checkout of that revision if needed, with:
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tasks/audits/2026-09-07_output-quality/diagnose.py`.
- [diagnostics.json](diagnostics.json): exact observed results and code/configuration.
- [loss_specs.json](loss_specs.json): all 31 canonical task specifications.
- [output_inventory.json](output_inventory.json): every tensor path in the inspected
  configuration, with shape, dtype and tensor-identity aliases.
- [historical_evidence.json](historical_evidence.json): copied existing evidence with
  source paths and SHA-256 hashes. This is an audit snapshot, not a second experiment registry.

The inventory uses expanded topology, d_model 32, two layers, four heads and
default output settings, with batch size one. It contains 172 tensor paths and
144 distinct tensor objects. These are configuration-specific implementation
counts, including diagnostics; neither number is an assay count. Equal-valued
separately computed aliases may not share tensor identity.

Independent regression checks: **136 passed**, covering data support, parameter
gradients, holdout collection/parity, model I/O routing, canonical loss/MIL and
collation. Two existing PyTorch scheduler deprecation warnings were emitted.
The initial diagnostic fixture failed the minimum MHC groove-length validator;
it was replaced with valid-length artificial groove segments before the final
diagnostic run. Production source and tests were not changed.

## Output-to-supervision matrix

All base weights below refer to the canonical task registry. The default
real-data CLI uses task-mean aggregation and learned uncertainty weighting;
sample-weighted aggregation is optional. Regularization is added separately.
The number of supported targets is therefore not the effective loss weight.

| Output family | Label and objective | Support/interpretation limits |
|---|---|---|
| `assays.KD_nM`, `IC50_nM`, `EC50_nM` | BindingRecord values, log10(nM), qualifier-aware loss; generic `binding` and family-specific tasks, weight 1 each | Generic binding sends every binding-family observation to KD as well; direct KD, IC50, EC50 and proxy data must be separated for quality claims |
| `KD_proxy_ic50_nM`, `KD_proxy_ec50_nM` | Aliases of KD in default merged grouping | Separate proxy targets are collated but have no dedicated canonical `LOSS_TASK_SPECS` entries. Optional split-proxy/leaf configurations need their own routing/support audit; specialized historical trainers are not proof for the canonical trainer |
| `assays.kon`, `koff` | KineticsRecord rates, log10(rate), MSE, weight 1 | Source qualifiers exist; consistent kinetic censoring is still missing. Sparse labels and nonzero gradients do not establish kinetic fidelity |
| `assays.t_half`, `Tm` | StabilityRecord; hours converted to log10(minutes), temperature to `(C - 50) / 15`; censor-aware, weight 1 | Physical-unit conversions are coherent in the inspected code. Per-family and per-method support may be very sparse |
| Binding descriptor panels: 26 columns over type/prep/geometry/readout | Gather observed column on binding rows; log10 affinity target and qualifier; mean across axes, base weight 1 | T_HALF/TM/KOFF/KON vocabulary entries do not receive the corresponding quantitative family labels through this binding-only loss. Merged source loses known prep/readout selectors; unknown columns are trained instead |
| `binding_affinity_probe_kd` | Generic binding labels, qualifier-aware, weight 1 | Auxiliary score supervision, not an additional measured endpoint or independent validation set |
| `elution_logit`, `presentation_logit`, `ms_logit` | ElutionRecord detections; MIL Noisy-OR bag BCE | Presentation is a proxy supervised by the elution label. `ms_logit` aliases elution; MIL nevertheless adds both `elution` and `ms` loss terms. This multiplicity must be explicit when interpreting weights and loss curves |
| `tcell_logit`, immunogenicity outputs | TCellRecord responses; row BCE or T-cell MIL, weight 1 | Shared response labels do not identify distinct biological latent quantities; recognition is trained indirectly through upstream coupling |
| T-cell panels: 56 columns over six axes | Selected-column response BCE, weight 1 per supported axis | Unknown columns masked from axis objectives; pathway-MIL rows excluded from all six axis objectives with no replacement panel loss |
| `processing_logit` | ProcessingRecord labels, BCE, weight 1 | Latest merged source scan had zero real processing records; Hitlist adapter returns none. Synthetic processing labels and downstream gradients are different evidence |
| `excision_logit` and component scores | Optional bulk-MS observed peptides and generated wrong-enzyme negatives; BCE, weight 1; downstream elution coupling | Component scores lack independent component-level labels. Mechanistic identifiability is unproved |
| Excision APM/stimulus panels: 8 + 7 columns | Observed condition column against elution response; base weight 1 | Elution is a proxy for excision. Context-column support/balance and causal interpretation are not established |
| `ms_detectability_logit` | Optional bulk-MS graded detection-depth labels, BCE, weight 0.5 | Soft proxy targets; thresholded binary metrics alone do not measure agreement with graded values; abundance/acquisition confounding remains |
| `tcr_evidence_logit`, three method outputs | Receptor database evidence; BCE, weights 0.05 and 0.02 | Canonical merged adapter sets evidence positive and supplies no method strings/bins. Presence-only evidence cannot establish receptor-specific matching or real-negative discrimination |
| MHC class/species/fine-type outputs | Derived annotations; CE, weight 0.1 | Training-only target resolvers are absent from split-support enumeration; CE outputs get no held-out rows. Host/molecular species and partial-chain label semantics remain #46 work |
| Antigen origin and foreignness | Source organism category and derived foreignness label; CE/BCE, weight 1 | Organism-derived proxy is not validated repertoire-relative self/nonself ground truth; origin CE is omitted from held-out rows |
| Core/register outputs | Binding gradients plus optional `core_start` CE | Tests explicitly exempt `core_start` because no source record supplies it; latent/register accuracy is not established by binding loss |
| Segment/interaction/latent vectors, probabilities, aliases, attention and contribution diagnostics | Upstream losses, deterministic transforms, aliases, or diagnostic quantities | Do not demand a separate biological label for every tensor or count aliases as independently validated endpoints |

## Verified coverage and ingestion gaps

`audit_split_support` iterates `batch.target_masks`, not the complete effective
loss registry. It misses dynamically derived MHC targets, bag objectives and
ad hoc panel losses, and aggregates T-cell panels by axis rather than column.
Its binary-target list omits all six newly binary T-cell axis tasks. It also
does not separate real labels from synthetic/proxy support. `require_all_active`
only requires targets seen somewhere; entirely absent declared outputs escape.

Reproduction: each split contains an ELISPOT positive and an ICS negative.
Both all-active gates pass even though each observed method column is one-class
and most columns have zero examples. A separate T-cell MIL fixture is actively
trained but contributes no response support to the audit.

The parameter test sums gradients over whole named tensors. An embedding table
with one trained row satisfies that check while other column-specific rows may
never be selected. Tests use constructed modality coverage, not the production
corpus, split distribution or effective update counts.

The merged adapter reads `assay_type`/`assay_method` but omits both when creating
BindingRecord. A supplied `purified MHC/direct/fluorescence` method becomes
`None`; prep, geometry, readout and method selectors all become index 0. This
is a real input shape: local TSV line 6 contains EVMPVSMAK, HLA-A*03:01,
473 nM KD (~EC50), exact qualifier, and that method. This is one observed
source example, not a measurement of affected corpus prevalence.

For T-cell pathway MIL, row masks correctly prevent assigning a bag response
to every molecule. The missing counterpart is selected-column bag supervision:
only scalar T-cell/immunogenicity outputs are consumed by MIL. In the diagnostic,
the ELISPOT embedding row receives exactly zero gradient while the unknown row
receives nonzero gradient and `tcell_mil` has an active loss.

## Verified evaluation gaps

- Two elution instances at probability 0.5 produce training bag probability
  0.75 and loss 0.287682. The held-out collector writes probability 0.5 from
  its independent row forward. Shared row inputs do not provide bag parity.
- T-cell MIL responses disappear from per-example dumps because their row
  masks are off and the collector never visits the bag channel.
- A supervised MHC-class example yields zero CE prediction rows: logits are
  flattened and skipped when their length differs from scalar targets.
- A three-component receptor-evidence target is flattened into three records;
  only the first retains the sample ID/source, and none identifies the output
  component. Larger batches can misattribute one sample's component to another.
- Binding/excision condition panels are outside the collector's task registry.
  Selected T-cell rows are dumped, but the output column/selector identity is
  not retained, preventing per-column reconstruction from the CSV alone.
- Final held-out loss disables the batch-count cap but still receives
  `max_mil_instances` (CLI default 128). Complete, reproducible bag evaluation
  therefore needs its own uncapped/chunked contract.
- `auprc` processes tied scores one row at a time. Labels [1,0] with equal
  scores return 1.0; reversed labels return 0.5. Threshold-based average
  precision for either constant predictor is prevalence, 0.5. AUROC already
  averages tied ranks and does not have this particular defect.

## What the existing real-data evidence establishes

| Evidence | Scope and result | Limit on present claims |
|---|---|---|
| 2026-09-05 PR #45 merged preflight, c4dc1cf | Scanned 3,266,972 TSV rows; before-cap routed records: binding 249,292; kinetics 106; stability 12,259; processing 0; elution 2,630,813; T-cell 207,987; receptor evidence 166,285 | Historical source routing counts; not current post-curation per-column/split support |
| Same merged preflight, capped and MHC-filtered | 287 samples; 173/57/57 split. EC50 support 1/0/0, Tm 1/0/1, receptor evidence 13/3/6 all positive; no method-bin support | Broad record-type ingestion does not ensure evaluable output coverage. This condition did not train |
| PR #45 Hitlist training smoke | 216 samples, 130/43/43 split, d32/l1/h4, one epoch/two train batches, no synthetics. Validation/test binding exact Spearman -0.204/-0.159, exact RMSE 1.621/1.628 log10(nM), threshold balanced accuracy 0.5/0.5 | Execution and checkpoint round-trip evidence, deliberately insufficient optimization for quality conclusions; old encoder |
| 2026-09-06 lineage/funnel closure | Data preflight only, no optimization or predictions | Does not extend model-quality evidence |
| September 2–3 training families | Historical binding learning exists; mapping-policy conclusions carry a null-to-NAN validity erratum, and generated decoys/one-class endpoints constrain other claims | Do not transplant these scores to the new encoder or call them all-output validation |
| September 4 groove-corrected replacement | Registered README says GPU conditions not launched; CPU preflights passed | No replacement predictive result |

The current Hitlist adapter intentionally returns no processing, T-cell or
receptor-evidence records. Adding synthetic cascades does not supply measured
responses for those missing modalities. Default merged input is broader but
loses context/lineage and has the binding metadata defect above. No single
source mode currently proves the complete intended corpus contract.

## Required fresh quality evidence

1. Freeze code and dependencies, exact curation, per-output support and separate
   data/split seeds. Repair measurement gaps before a multi-output quality claim.
2. First establish fitting on a declared, representative real training subset,
   with no evaluation leakage. Save initialization and learned per-task losses,
   effective weights/gradients and predictions; compare to simple baselines.
3. Run the canonical trainer from fresh compatible weights with adequate
   optimization, peptide-disjoint train/validation/test and validation-only
   selection. Diagnose cross-source duplicates and synthetic-parent overlap.
4. Preserve full selected-checkpoint bag/row predictions and task-specific
   validation/test metrics. Separate assay families, exact versus censored,
   real versus generated, class/allele/source/context and supported columns.
5. Join known examples to actual observation IDs, measurement family, value,
   qualifier, unit and split. Existing SLLQHLIGL probes save predicted KD and
   probabilities, but no observed targets or membership. Inter-allele variance
   is a sensitivity diagnostic, not a correctness score.
6. Report exact-value Spearman/Pearson/RMSE, qualifier-aware 500 nM binding
   metrics, binary discrimination and calibration where justified, categorical
   metrics and soft-target agreement where applicable. Declare sample sizes
   and unavailable metrics rather than manufacturing balance or thresholds.
7. Check documented aliases, conversions and algebraic invariants; review
   contradictory examples in their actual assay/context. Set prospective
   per-endpoint criteria after examining support; test data must not tune them.
8. Register the training experiment and preserve all required artifacts in
   `experiments/`. No inference about biological validity follows solely from
   finite loss, a green suite, a named latent or a loss decreasing in aggregate.

## runplz follow-up

runplz 4.4.4 is published and the local editable runtime reports 4.4.4, while
installed distribution metadata still reports 3.24.31. Its SSH/Brev lifecycle
reliability improvements are useful for training and inference orchestration.
For long Modal runs, [runplz #165](https://github.com/pirl-unc/runplz/issues/165)
still tracks supported detached execution and later artifact collection.
Large dumps should use a Volume, avoiding the 256 MB return-archive limit.
Pin Presto's explicit H100! hardware contract and record actual hardware;
resource minima alone need not select that GPU. No runplz code was changed,
package upgraded, provider job launched, or new runplz issue filed.

## Filed issues

All six issues were published and their titles, complete bodies and open state
were read back and verified. [Publication receipt](filed_issues.json).

| Issue | Scope |
|---|---|
| [#48](https://github.com/pirl-unc/presto/issues/48) | Audit and enforce supervision coverage per output column, source, and split |
| [#49](https://github.com/pirl-unc/presto/issues/49) | Preserve binding assay descriptors when loading merged TSV records |
| [#50](https://github.com/pirl-unc/presto/issues/50) | Supervise selected T-cell assay response columns on pathway-MIL bags |
| [#51](https://github.com/pirl-unc/presto/issues/51) | Export held-out predictions for the same bags, panels, and vector outputs that training supervises |
| [#52](https://github.com/pirl-unc/presto/issues/52) | Fix average precision: tied scores currently make results depend on row order |
| [#53](https://github.com/pirl-unc/presto/issues/53) | Establish a fresh post-#47 real-data baseline with measured examples and complete validation/test evidence |

Issue #53 links the preceding five as prerequisites for a complete multi-output
quality claim. The issue bodies retain pinned source evidence, reproductions
and acceptance criteria. This audit bundle was preserved in the first repair
PR; the GitHub issues are self-contained and do not depend on this report.
