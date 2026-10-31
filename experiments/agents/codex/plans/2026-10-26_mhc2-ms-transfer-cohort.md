# MHC-II MS source-policy and transfer-cohort qualification

Codex / GPT-6, 2026-10-26. Full objective remains
`tasks/mhc2_improvement_spec.md`. Family:
`experiments/2026-10-26_1954_codex_mhc2-ms-transfer-cohort/`.

## Decision to resolve

The preceding verified admission retains98,220reported positive MS detections
with complete source candidate pairs. That is evidence admission, not yet a
training cohort. Before fitting, distinguish experimentally supported restrictions
from prediction-derived assignments, preserve probable-background qualifications,
and establish source/sequence dependencies for an actual transfer comparison.
Do not change the earlier admitted file or active binding runtime.

## Inputs and literal audit

- Frozen admission JSONL and contributor summary/links from
  `/private/tmp/presto-mhc2-20261026/ms-admitted/attempt01/`; assert hashes from
  the independently verified preceding family before reading rows.
- Frozen human native source, SHA256
  `f902341e90c1c57defc93f4bc540ffcbea1b41061afb4492ddd8aaa7642c65bd`.
  Native producer remains Hitlist1.78.0; current curation is separately pinned
  Hitlist2.0.42, commit13b91424295e01286db46e8e9818fcc2fe816834.
- Current `pmid_overrides.yaml` and source revision metadata: extract only the
 20retained studies, retaining all literal notes, rules and sample qualifications.
  Do not backfill current arm claims onto older rows without a source join.
- Original IC50 train/validation/test sources, already pinned by the1929family,
  supply an initializer-exposure inventory only; inspect no new predictions.
- Stream each admitted row once to count joint source/molecule/restriction/
  condition policies and retain compact source-linked metadata. Join every
  contributor PMID. Compute exact peptide and any contiguous9-mer overlap among
  studies and against the IC50 initializer's training inputs. This measures only
  those dependencies; absent overlap does not establish parent/homology, donor,
  specimen, search or teacher independence.

## Primary evidence review

Review original Methods for the six largest retained studies: PMIDs35251023,
33858848,25502872,30573663,33592498,34433824. Source selection is by existing
admitted row counts, before model outcomes. Resolve whether retained ligand lists
or allele assignments were predictor-filtered, what pair/capture evidence exists,
and any explicit background-control qualification. Start with current source notes
and primary publisher/PMC records. Bound this stage to16network discovery/open
operations across tools and20minutes of retrieval; preserve failed retrievals and
do not infer unavailable Methods. Smaller-study policies remain unknown unless
already supported by reviewed primary evidence.

Prospective retrieval amendment, after the first16operations: six PMC opens
returned browser challenges and the Atlas publisher returned403; two publisher
pages were retrieved but their initial text windows omit Methods. Allow at most
24total operations, preserving the original16and20-minute window, for Methods
windows on those two pages and public EuropePMC XML retrieval for the remaining
four already selected papers. This does not change study selection or admission
criteria. Stop retrieval if this amended budget is exhausted.

The CIITA source's parental identifications are explicitly described as probable
background in current curator notes. Preserve this qualification. Do not label
all parent rows negative or claim a causal route effect from identification counts.
Missing donor/aliquot/parent identities stay missing. If the current exporter drops
a material qualification, reproduce that precise gap and check existing issue
ownership before filing or adding evidence.

## Transfer cohort decision before training

Use the source-policy/overlap evidence to freeze a separate predictive recipe.
An eligible source must have defensible measured MS detections and complete
candidate scope; predictor-filtered/assigned evidence cannot silently serve as
independent binding or ligand validation. Ambiguous mixtures remain candidate
sets, never a best-predicted allele. No test outcomes or model predictions enter
source selection, split construction or control generation.

The planned predictive estimand is reported MS-detection enrichment relative to a
declared generated sequence background, unless genuine measured negatives become
available. Generated controls must remain explicitly generated, retain matched
length/composition/sample conditions, and be excluded if already observed in the
full source. They are not measured nondetections. The exact control distribution,
split/study membership,9-mer exclusions, initializer exposure policy, model arms,
update budgets, metrics and acceptance gates must be frozen in the next predictive
plan before any scoring or fitting. Compare binding-only and sequence/context
controls, with true source transfer where qualified. Preserve source dependence
and conditional-background limitations even if discrimination is strong.

## Resources, artifacts and verification

Local CPU1thread, no GPU/cloud/install. Guard each scan at300seconds/2GiB RSS,
availableRAM>=10GiB and free disk>=10GiB; generated compact artifacts<=250MiB.
Do not interfere with the active old IC50/species or new censor-refinement jobs.
Record actual resource use, source/code hashes, git dirty state and exact invocation.

Publish source-linked policy tables, compact metadata, overlap summaries and
literal primary-evidence references in this family. Independently replay row
counts/source joins and overlap on original inputs before choosing the cohort.
No model is fitted in this qualification family, so training/validation/test loss,
binding metrics and prediction dumps are inapplicable; the next predictive family
must produce all of them. Add the completed decision to `experiment_log.md` and
advance to the qualified MS fit, rather than treating this audit as the goal.

- [x] Freeze and stream the policy/dependency inventory.
- [x] Review primary restriction/background evidence and existing issue ownership.
- [x] Independently verify counts and overlap.
- [ ] Freeze an eligible predictive cohort/recipe and proceed to actual MS fitting.
