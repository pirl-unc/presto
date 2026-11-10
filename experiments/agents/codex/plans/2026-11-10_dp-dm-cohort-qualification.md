# Qualify DP/DM cohort identities before condition-aware training

Date: 2026-11-10. Agent: Codex / GPT-6. Presto base 4da30f74.
Prior turn made progress: published Hitlist PR898 / Presto PR156 with native
source integration, independent partial overlap replay and explicit open gates.

## Intended result

Determine the actual DP/DM sample/allele/context support after the complete known
sequence embargo, rather than treating the previous 13,209 partial-clear rows
as trainable. This is an identity audit, not a biological performance experiment.
It leaves the active binding, MS and processing objectives intact.

## Frozen inputs and access boundary

- Reuse the preceding family's exact 21,547-row shared export and its pinned
  presentation and Atlas identity indexes. No new source threshold or label.
- Add the original presentation `held-sequence-embargo.txt.gz` and
  `reserved-study-closure.json` from the independently verified paired-v1
  cohort. This includes all reserved component peptides and final held sampled
  comparisons, not merely the already-consumed validation rows. Hashes:
  `1cfe25cfc6f3d636e0a49f1359b4eafb5fdf39ea7ec52a541629905c3c68ce57`
  and `2b3a40b53fd5c73de3ad45ae2bf6fb3ac83eee651087489ce5310264faad303d`.
  Never open the locked confirmation observations or outcomes.
- Reuse the seven model namespaces pinned in the prior affinity identity audit:
  Stage-B TRAIN/development, source-review-v2 TRAIN/validation/previously consumed
  TEST identities, and NetMHCpan42/NetMHCIIpan43 BA TRAIN. Extract only top-level
  peptide, observation_id, study_id, group_id, original_measurement_id, source,
  pmid and reference_iri using the already-tested allowlist JSON projector.
  Hash all input bytes, but do not deserialize/retain labels, values, predictions,
  target statistics or nested provenance. Never mount/read locked confirmation
  records. Distinguish existing TRAIN overlap from protected evaluation overlap.
- Reuse T-cell split `row-membership.jsonl.gz` and embargo-drop identities under
  the independently verified split manifest hash
  `12aa0fe921f89cf44673040943e2a5dfdb4b0c8918d0d9e0d585befc39cade0d`.
  Project only peptide, partition/base_partition and component/assay identity.
  Keep dropped protected-fold identities in the embargo; no outcomes or scores.

Freeze every path/size/hash in launch.json before executing. Schema/path discovery
may read identity-only manifests; mixed data may be accessed only through the
explicit field projector. Stop on a changed hash or unsupported schema, preserve
failed output and amend the plan before retrying.

## Computation and verification

1. Create reusable outcome-free affinity identity files on the existing Modal
   volume. CPU only, one container, 8 CPUs / 16 GiB; no GPU. Independent file
   scans may run concurrently. Persist row counts, input/output hashes, exact
   runtime and software; return only compact receipts. Do not refresh sources.
2. Compute exact, shared contiguous 8-mer and same-length Hamming <=1 overlap
   for every namespace. Hamming identity uses same-position deletion masks
   `(length, position, peptide with that position deleted)` as in the common
   data/netmhc_training.py; never compare deletion strings without position.
3. Source component aliases PXD017154 / PMID33025005 must not coincide with
   any protected study closure. Candidate sample A/B labels do not establish
   independent specimens; all samples share one study origin.
4. Build per-row reason flags and union embargo, preserving the original source
   cohort. Report per-sample, exact pair and DM state attrition, overlap of
   nonembargoed peptide identities across conditions, and prospective internal
   comparison support. No TRAIN eligibility until all listed namespaces pass.
5. Independently replay row membership with direct one-substitution enumeration
   against reference peptides (separate from the deletion-mask implementation)
   and a separately constructed core reverse index. Include fixtures proving
   ignored outcome fields cannot alter the identity projection and that shifted
   deletions cannot masquerade as same-position Hamming neighbors.
6. Freeze a follow-on scientific comparison only if data retain usable paired
   support. Prefer the shared MS/context model; do not manufacture affinity,
   abundance or pathway labels from detections/nondetections. The new source is
   one engineered cell line and is not an independent generalization benchmark.

## Closure

Family: experiments/2026-11-10_1607_codex_dp-dm-cohort-qualification.
Commit README, source/launch snapshots, summary tables/JSON, independent replay
and durable artifact receipts; update experiment_log.md and Presto155. Predictive
validation/test metrics and dumps are inapplicable: no model fit or held-out
outcome use occurs. No release merge or model-improvement claim follows from
admission. Other Atlas/T-cell work remains owned by the independent worktree.

Pre-execution source-identity detail: include the same frozen T-cell split's
`component-membership.jsonl.gz`, projecting only component_id, base_partition,
pmid and doi. Row membership alone omits PMID, so it cannot prove absence of a
shared source study. A protected component matching PMID33025005 embargoes the
entire DP/DM source. Preserve this check independently of peptide overlap.

## Pre-fit structural check (frozen before fixture execution)

Code inspection finds that `affinity_downstream` computes its context-sensitive
term from full peptide + APC/source context, without the MHC representation,
then adds context-independent affinity/stability. Probe the interaction capacity
before proposing a fit: two artificial MHC identities in each class, three
synthetic canonical peptides per class, seeds42/43, both existing staging modes,
DM present/absent, dropout0/eval. Preserve logits, intervention deltas, register
weight deltas, and purified-affinity invariance. No learned checkpoint, optimizer,
real biological target or accuracy metric. This distinguishes feature sensitivity
from the ability to learn MHC-dependent editing. A structural limitation should
be recorded in a focused issue and the next shared-model comparison plan; do not
silently reinterpret a supply-side processing term as measured peptide editing.
