# Macrophage polarization composition admission

Codex / GPT-6, October 24, 2026. **Complete: prospective support screen failed; no model trained.** Registered at Presto 534ae42c2808a74199d3090ba48303c1a30f6cd5 with existing task dirt and frozen experiment-local code. The preceding 1702 MS1 observer remains frozen.

## Question and fixed contract

Can canonical PXD057588 Tables 7/8 support a fresh within-parent, within-table/channel positive TMT-intensity composition comparison? A static peptide-specific MS response cancels from same-peptide treatment/control ratios, so these compositions were considered for a future transfer test. They remain MHC-selected measured signal, not intracellular peptide concentration.

Use `/private/tmp/hitlist-v2-reference-200-final`, manifest `d99f67e625c6c33a2054a56dea4f98fb8248ca5f717cc98af3662f228655fa6a`, with released Hitlist 2.0.26 at `/private/tmp/hitlist-release-2026`. Preserve raw positive finite `raw_tmt_intensity`, all source rows/cells, missingness, alternative origins, protocol and material-link uncertainty. Require canonical sequence and one full-parent sequence under the pinned reference; identical-sequence accession aliases are not multiple parents. Exclude all components linked to any parent/component used in the previous Table 9 predictive family, including its training set. No previous outcome file was used.

Group by parent and original table/channel, requiring at least three distinct eligible peptides. Split whole components using the frozen SHA256 salt and 70/15/15 buckets described in `reproduce/plan.md`. Require at least 100 training, 25 validation and 25 test parents. No new salt, eligibility relaxation, missing-value imputation, invented negatives or synthetic biological labels.

## Outcome

| Split | Distinct supported parents/components | Supported channel groups | Required parents | Decision |
|---|---:|---:|---:|---|
| Train |45|324|100|Fail|
| Validation |5|30|25|Fail|
| Test |7|54|25|Fail|

There are 57 supported parents, not 408 independent examples represented by the repeated channels. Inventory covers 6,378 source rows and 38,268 raw quantity cells. Before the group-support minimum, 2,760 rows remain eligible. Exclusion reasons include prior component/parent overlap (2,111 rows), ambiguous full-parent identity (2,264), and noncanonical parent sequence (6); these reasons overlap and must not be summed as unique exclusions. Prior coverage includes 512 components and 920 full-parent sequences.

All three support gates fail. This cohort is not admitted for the proposed predictive comparison. Existing predictive baselines are unchanged; no new biological accuracy, concentration estimate or observer transfer is claimed. Do not fit this design by relaxing its thresholds after seeing coverage.

## Measurement limits

Primary-study context is recorded in `results/primary-context.md`: TMT chemistry and aggregation of oxidized/nonoxidized forms differ from the independently fitted unmodified MS1 observer. Shared study controls, unresolved exact acquisition/specimen/form links, source search versus mapping reference generations, and conditional positive-only support remain explicit. The original Europe PMC XML download returned HTTP 500; primary methods were inspected through indexed primary-paper text. This was not a successful XML download. Public-server Set-Cookie headers are omitted from publication derivatives; originals remain in `data/apc_composition_admission/2026-10-24_1755_v1`.

## Verification, reproduction and resources

`reproduce/launch.sh`, `launch.json`, `plan.md` and `source/` preserve the exact launch, environment and executed sources. Local CPU, one numerical thread; Python `/Users/iskander/code/shared-virtual-env/bin/python`; no Modal, GPU or shared-environment changes. Raw source inputs stay at the canonical package path. Results retain source hashes/schemas, every row/quantity disposition, parent/origin/component exclusions, channels/context assertions and candidate group membership.

Inventory: 17.05 seconds, 0.948 GiB peak RSS, 3,055,841 new artifact bytes and 4.71 GB free disk at completion. Independent verification: 1.02 seconds, 0.474 GiB; separately decoded original OOXML, reconstructed the component/parent graph by breadth-first traversal, and verified all 38,268 quantities, 12,756 sequence/parent cells, 6,378 dispositions and 13,770 candidate groups. All joins, prospective splits and failed support gates reproduce.

The initial verifier failed because integral Excel numeric headers became `126.0` instead of canonical text `126`. Its executed source and log are preserved in `results/verification-attempt1/`. `reproduce/verification-recovery.md` freezes the rendering-only correction; quantities, admission, splits and thresholds did not change. Corrected code passes isolated Ruff, and `results/independent-verification.json` records success.

This is source admission: no training/pretraining, optimizer, loss, assay head, synthetic target, validation/test prediction dump or predictive metric applies. The candidate split was inspected for coverage only. Full APC availability/concentration prediction remains unachieved.
