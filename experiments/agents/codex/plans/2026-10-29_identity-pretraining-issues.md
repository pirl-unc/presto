# Identity pretraining and interpretable conditioning: experiment backlog

2026-10-29; Codex / GPT-6. User requested assessment and issues for later
experiments, not a model change or a training launch in this task.

## Scope and plan

- [x] Inspect the canonical encoder, identity heads, resolver, current fit config,
  existing issues #46/#115/#123/#130 and prior reference experiments.
- [x] Check primary research on TCR representation pretraining, paired-chain
  lineage prediction and user-editable concept models.
- [x] File focused follow-up experiments covering all seven suggested labels and
  the proposed two-stage inference, linking existing broad requirements.
- [x] Verify published issue bodies and links; record priorities in the backlog.

## Current evidence

At base commit 858f5383094ad1a8e840ddb1b4df8233b189b2df with an already dirty
working tree, `models/presto.py` is canonical; `models/modular.py` is an alias.
The shared encoder already supports optional MHC class/species, TCR species,
CD4/CD8 lineage, MHC/TCR pairing and peptide-reference-membership heads.
`data/modular_context.py` resolves override > supplied > predicted values while
retaining provenance. Identity probabilities are detached from downstream assay
gradients; encoder representations also feed downstream prediction directly.
The current temporal ten-output fit has empty `identity_tasks` and `species`,
and identity weight zero. Existing plumbing therefore does not establish
pretraining benefit or qualified identity prediction for that checkpoint.

The 2026-10-21 Weirdo eight-mer reference experiment failed its prespecified
comparison against amino-acid composition; it did not test downstream transfer.
The 2026-10-26 MHC-species registry entry records verified controls and unfinished
neural conditions. Audit its artifacts before reusing it, rather than reporting
a successful neural species model.

## Issue boundaries and priorities

1. Shared molecular metadata pretraining: MHC/TCR species, MHC class and actual
   chain/domain identity, including TRA/TRB/TRG/TRD and receptor alpha-beta versus
   gamma-delta type. High-confidence labels are cheap; transfer is unproven.
2. MHC pair consistency: separate typed role compatibility, reference provenance
   and experimentally demonstrated assembly. A shuffled pair is not a measured
   nonassembling molecule. Prioritize an assay-transfer comparison.
3. Phenotyped-cell CD4/CD8 supervision: keep phenotype separate from receptor
   family and candidate restriction; donor/receptor-cluster holdouts are needed.
4. Peptide/protein taxonomic membership: lower priority, retain multiple origins,
   compare against composition/reference lookup and prior failed screen.
5. Conditioning comparison: auxiliary-only shared encoder versus detached soft
   identity context versus a hard-label cascade. Retain sequence representations,
   confidence, explicit provenance and per-field override semantics. Test missing
   metadata and interventions instead of assuming the two-stage design helps.

## Common experimental requirements

Use one observation manifest, trainer and shared encoder; no task-specific model
forks. Classifier labels cannot leak through input names, slots, copied context,
source identifiers or candidate MHC class. Label availability, downstream input
availability and biological truth are distinct. Register eventual experiments
with frozen sources, eligible targets, paired seeds, matched assay exposures,
compute accounting and independently verified per-example validation/test results.
Measure auxiliary calibration and downstream affinity Spearman/Pearson/RMSE,
500 nM decisions, and source-scoped ligand AUROC/AUPRC where supported. Keep
generated comparisons distinct from measured nonresponses. Auxiliary accuracy
alone is not a promotion criterion; preserve null/negative results.

No data acquisition, model change, GPU launch or completed experimental result is
claimed by this issue-filing task. Existing curation work and artifacts remain
unchanged.

## Published and verified

- [#133](https://github.com/pirl-unc/presto/issues/133): molecular species,
  MHC class, chain/domain identity and alpha-beta/gamma-delta receptor family.
- [#134](https://github.com/pirl-unc/presto/issues/134): MHC pair consistency,
  keeping reference controls separate from experimental incompatibility.
- [#135](https://github.com/pirl-unc/presto/issues/135): phenotyped CD4/CD8
  supervision and paired-receptor transfer.
- [#136](https://github.com/pirl-unc/presto/issues/136): ambiguous source taxonomy
  and protein-context pretraining, retaining the prior negative screen.
- [#137](https://github.com/pirl-unc/presto/issues/137): auxiliary-only versus
  soft identity conditioning and hard-cascade controls, with manual overrides.

All five issue bodies were read back through GitHub and compared exactly with
the reviewed drafts; all are open. This completes assessment and issue filing,
not the proposed experiments. No model, dataset or active fit was changed.
