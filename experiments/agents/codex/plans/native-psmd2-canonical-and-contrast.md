# Native PSMD2 perturbation: canonical source, then precursor-cancelling contrasts

Follow0344, which recovers the original Dataset7 intracellular panel and its six
run identities. Follow0143's qualified but uncertain sequence development gain;
retain all earlier exact/sampled failures and original held-out closures. The
full goal remains precursor-relative APC peptide availability, not an observed-
set or response-only score. This new human HEK293T panel is a useful native-cell
constraint, not an APC concentration validation.

## Source and canonical admission before modeling

Use only the exact original Sahu2021/PXD018722 Hi20S panel initially. Source audit
0344 pins the recovered workbook, literal cells, public project/file metadata,
native experimental design and MaxQuant configuration. Dataset6 is identity-only
and does not enter an abundance loss. Hypoxia, heart and proteasome-trapped panels
remain separate potential future populations; do not pool their different capture
and processing methods. Preserve every source row and excluded/deferred reason.

Implement in an isolated Hitlist feature worktree based on current authoritative
main, using the existing standalone evidence/processing and recipe interfaces.
Do not create a second Presto curator. Reuse original-file and material/context
contracts. Add regression fixtures covering date-shaped gene names, raw zero,
matching-only identifications, all parent alternatives, search-length discrepancy,
sample mappings and incomplete source assets. Follow Hitlist's required version,
format/lint/full-suite/PR/merge/tested-release workflow; no shared environment
installation, Modal training or alteration of other worktrees.

Recover the exact deposited reference (33,003,906 bytes), with all sequence
alternatives and distinct human/reference-reporter provenance. Retain unsupported
reporter records in source evidence; prediction candidates are endogenous human
products. Do not replace the original reference with a current canonical isoform.
If accessing original processed `Hi20S_txt.zip` is necessary to establish quantity
aggregation or peptidoform coverage, prefer indexed range retrieval of its specific
small text members, preserving member CRCs/offsets and original central directory.
Do not download RAW/Andromeda payloads or the entire167MB archive by default.
This next, separately registered admission may use64MiB per source asset and256MiB
total, because the known33MB reference is a required input. The original0344 caps
and failed attempts remain unchanged. Keep1GiB continuous free reserve plus32MiB
startup margin,3GiB RSS and20minutes per stage. Full local Hitlist tests require2GiB
headroom. A capacity refusal defers dependent work rather than changing a guard.

Source distinctions required:

- Native intracellular peptide extraction versus MHC elution, proteasome capture,
  purified digest, standard proteomics and fluorescence.
- Human HEK293T species/cell context; inducible PSMD2 knockdown, reported source
  composition and actual measured activity as separate assertions. Do not set
  machinery activity to zero or use a published aggregate percentage per sample.
- Cont1–3/KD1–3 to native RAW identities. Preparation/clone independence and pairing
  remain unknown unless explicitly established. Doxycycline, reporter, inhibitors
  or handling fields must carry source/sample scope; no blanket defaults from
  the article's list of experiments.
- Source `Intensity`, total across samples, matching flags, PEP, source decoy and
  contaminant indicators. Do not rename intensity LFQ or call matching-only values
  MS/MS-confirmed. Sequence-level aggregation is not an unmodified form label.
- Deposited8–25 search support alongside the article's minimum7 statement. Shorter
  products are unsearched. Zero means source zero with unknown observation cause,
  not a measured absence or known censor threshold. No inferred negative labels.
- Preserve13 date-shaped gene annotations as raw serial/style/rendered-date
  evidence, not gene symbols. Retain all protein-accession alternatives.
- No matched precursor, recovery or absolute amount has yet been established;
  protein groups inferred from this peptidome are not independent precursor input.

## Scientific question for the first fit

Does loss of a19S component change the *relative abundance of products from the
same precursor* in a predictable sequence-dependent way? This is more informative
about processing than fitting a common control/KD offset, while avoiding an
assumption that the precursor protein abundance stays constant.

For two unambiguously same-parent peptides i and j measured in the same control
and KD acquisitions, the candidate target is:

`[log10 I(i,KD) - log10 I(j,KD)] - [log10 I(i,control) - log10 I(j,control)]`.

Under a multiplicative observation model, shared precursor abundance and run load
cancel within each parent, and peptide-specific fixed MS response cancels across
conditions. This does **not** prove cancellation of condition-dependent extraction,
ion suppression, modification composition or clearance. Check the source protocol
before adopting it. If those assumptions are unsupported, report a conditional
observed-signal contrast rather than claiming molar change.

Do not choose a reference peptide or conditions from the size/direction of these
effects. After canonical support is known, freeze a source-independent hash choice
of reference among eligible same-parent peptides, full dependency groups and all
shared-reference noise dependencies. Require sufficient distinct parent components
before fitting; define exact thresholds and splits prospectively in the fitting
registration. Preserve all alternative origins in the dependency graph before
eligibility; ambiguous precursor mixtures cannot be assigned to a preferred parent.
Repeated samples/peptides and overlapping sequences stay grouped. No old test is
reopened and no signal summaries determine model hyperparameters.

## Keep the candidate models small

First compare zero relative change, a length-dependent contrast and the previously
defined parsimonious sequence features (log length, composition, termini) under
matched training-only transforms and nested regularization. Prediction is
`g(features_i) - g(features_j)`. A common intercept cancels structurally, so do not
allocate or claim to train one. There is no peptide identity, predicted species,
MHC, TCR or irrelevant assay head in this subset. Supplied context is known human
and known perturbation; it does not train generalized species or machinery effects.

If the direct screen supports sequence signal, compare a shared physical
fragmentation model with the same observable contrast and an unchanged uniform
reference. Full fragment production, including unobserved short/nonbinding
products, remains represented. For a query span, exact expected fragment amount
can be evaluated without enumerating all spans; a discrete sampler must represent
the same complete partitions and initial-state/clearance assumptions. Steady-cell
data do not supply elapsed processing time: any inferred exposure is dimensionless
until independently calibrated, not a rate in minutes.

Register exact versus discrete objectives, optimizers, variance controls, seeds,
sample budgets and score/runtime accounting before that comparison. Preserve the
original REINFORCE/exact references. Explicitly distinguish exact-forward score
hybrids from a simulator-forward learner, and preserve any finite-sample bias in
log-of-expectation objectives. An RL extension should answer an empirical question
after the representation is useful, rather than duplicate a failed physical fit.

## Interpretation and required next evidence

This contrast can constrain precursor-relative *changes within the observed
intracellular pool*. It cannot identify each condition's complete absolute
distribution, missing short fragments, turnover/clearance rates or APC transfer
on its own. Couple only independently validated kinetics/observer components in
later jointly specified work; a static MS predictor cancels from this contrast and
must not be presented as a mechanism for improving it. The full availability model
still needs baseline distribution/quantity anchors and professional-APC validation.

Source admission must finish with independent original-cell/reference/run mapping,
all dispositions, common-schema metadata and a canonical report. A subsequent fit
requires its own timestamped registration, frozen exact support/weights/units,
validation and test prediction dumps, all regression/spread metrics, simple
comparators, meaningful gain criteria and uncertainty over independent dependency
units. No success claim based solely on numerical checks or a source importer.
