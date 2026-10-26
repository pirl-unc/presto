# Shared nontryptic calibration source package: original replay passes

The reusable Hitlist package now preserves the qualified PXD024917 source data
through public EvidenceStore queries. Independent verification passes for all
61,852 stored cells,348 S3 measurement slots,3,684 transition rows,684 transition
groups and54 reported form correspondences. This is source-integrity evidence;
no model was fitted and no biological prediction improvement is claimed.

## Source and curation contract

Codex/GPT-6,2026-10-26. Starting Presto commit155f719b85130367bff37f7d514c60dcb3a66502
and Hitlist0263dc639fdebe0a1532260f09cf575f99c4bf32 plus frozen dirty implementation.
Exactly ten original assets: S1/S3/S6/S7, two PRIDE transition exports, article XML,
Methods PDF and the previously retrieved project/inventory JSON. Original bytes
and hashes were qualified in families1816/1857. No new source retrieval occurred.

The source package contains nineteen typed relations, original bytes and protocol
assertions. Package schema29/catalog38; revisionpxd024917_nominal_ms2_v1. All hashes,
field schemas and typed rows are verified by replay on public open. A separate
verifier uses openpyxl and prior independently qualified cell/sum/mapping ledgers.
The compiler consumes original assets, never those derived verification ledgers.

-18 standards and72 reported isotope forms. S1 affinities remain predictions.
-116 nominal dose rows and348 response slots:339positive,9missing,0reported zeros.
-108 selected transition-count cells remain distinct from response intensities.
-54 export forms,684 groups,3,684 transitions and1,368 declared aggregate sums.
-1,017 form/dose candidate links preserve ambiguity;0inferred acquisition links.
-18 scoped protocol assertions and3material contexts/2links preserve spike stage,
  reported amount assignment, instrument, triggers and capture/aggregation scope.

S3A is free1H standard response after purification; S3B is embedded2H response
before immunoprecipitation. Neither labels native APC concentrations. The absent
low-dose row remains absent. C/D/E are source slots, not established acquisitions.
Source-reported signals and derived transition sums remain distinct because their
reconstruction is incomplete. S6A/S6B contradictions remain54conflicting forms,
17within the declared tolerance and1missing-residue assertion. No isotope-atom
repair or quantitatively convenient candidate matching is applied.

## Failure, diagnosis and correction

The initial compiler rejected a panel length disagreement. Original-source
inspection with two readers confirmed S1 E19 reports9 for the ten-residue sequence
AMLGTHTMEV. This annotation conflict is now queryable: calculated peptide_length,
reported raw/parsed length, source numeric status and length consistency status
are separate fields. The peptide and all associated observations are retained.
Seventeen other length annotations agree. Formula caches remain nonmeasurements.

The original executed source,request,failed log and staging package are preserved.
`reproduce/amendment-1.md` registers the diagnostic;`amendment-2.md` registers the
correction and extended independent check. The original verifier is untouched;
`verify_v2.py` adds only length-claim validation to the existing source checks.
The companion launcher adds its stage name without changing any resource guard.

| Stage | Result | Wall seconds | Peak child RSS bytes |
| --- | --- | ---: | ---: |
| verify attempt1 | Failed: overly strict length consistency |2.426955|274,776,064|
| inspect attempt1 | Original length claims independently replayed |0.568671|133,562,368|
| verify_v2 attempt1 | New package and all independent checks pass |6.169139|493,617,152|

All three stages launched; none violated a resource limit. Local CPU/one numerical
thread,Python3.12.6;3GiB startup free+speculative and process-tree RSS,20min/stage,
1GiB disk reserve plus32MiB startup margin,50ms sampling and OS child-peak check.
The proposed lower-memory amendment remains inactive. No Modal, installation or
shared-environment modification occurred.

## Artifacts and reproduction

- Original attempted code: `reproduce/source/`, exact request`reproduce/requests/verify-1.json`.
- Corrected Hitlist code: `reproduce/source-attempt2/`, exact request`reproduce/requests/verify-v2-1.json`.
- Results:`results/source-package-verification-v2.json`,`panel-length-diagnosis.json`,
  all stage runtime/process receipts and logs. Software logs are copied at closure.
- Immutable successful package:
  `data/nontryptic_calibration_package/2026-10-26_0236_codex_nontryptic-source-package/attempt2`
  (repo-relative; original/typed bytes remain outside Git).
- Package manifest SHA256:`ee304b6e9976135316e931cdcda4fd007d8f1e76c8a968c5253e2298e57ed876`.
- Failed staging:`data/nontryptic_calibration_package/2026-10-26_0236_codex_nontryptic-source-package/.attempt1-9di7m08u`.
- Original assets remain under `data/nontryptic_calibration_sources/2026-10-25_1816_codex_nontryptic-calibration-screen/`
  and `data/nontryptic_transition_sources/2026-10-25_1857_codex_nontryptic-transition-lineage/`.

Invocations from the worktree root (immutable outputs refuse replacement):

```sh
experiments/2026-10-26_0236_codex_nontryptic-source-package/reproduce/launch.sh verify --attempt 1 --requests experiments/2026-10-26_0236_codex_nontryptic-source-package/reproduce/requests/verify-1.json
experiments/2026-10-26_0236_codex_nontryptic-source-package/reproduce/launch.sh inspect --attempt 1 --requests experiments/2026-10-26_0236_codex_nontryptic-source-package/reproduce/requests/inspect-1.json
experiments/2026-10-26_0236_codex_nontryptic-source-package/reproduce/launch_v2.sh verify_v2 --attempt 1 --requests experiments/2026-10-26_0236_codex_nontryptic-source-package/reproduce/requests/verify-v2-1.json
```

For replay, register a fresh family/output location with identical source bytes;
do not overwrite these executed requests or snapshots. Environment overrides and
exact source/input hashes are frozen in the launch configuration and requests.

## Software and scientific decision

After the length fix, 63 focused tests and all 3,649 default-suite tests pass
(392 warnings, 329.69 seconds, 909,312,000 bytes peak RSS); format/lint pass.
The earlier 3,642-test pass and every failed attempt remain in the copied logs.
Shared implementation is published in Hitlist PR850 at commit
`ac3e04841e5a8b0655b1bbd91df2ffd3e23d8cc0`; all frozen implementation bytes independently match that Git tree.
Exact-head hosted integration/CI remains pending; no local integration is claimed. Synthetic fixtures exercise
source semantics and corruption; they are not biological training examples.

No training/pretraining,synthetic biological labels,loss terms or assay-to-model
outputs apply. No predictive split or per-example predictions exist because this
family qualifies source integrity, not a predictive model. Binding/500nM metrics
are inapplicable. No earlier biological held-outs were reopened and no model is
promoted. Source curation clears one prerequisite for a separate response fit.

Status: source qualification closed; shared implementation published, hosted CI pending.
Next step: the separately prospective [sequence-response diagnostic](../agents/codex/plans/nontryptic-sequence-response.md)
uses a shared declarative recipe, precursor-grouped folds and dose-only baselines.
Open questions: sequence/protocol transfer, native observation calibration,
machinery-conditioned production/survival, short/unobserved products and the
matched differentiable/discrete biological comparison. Full precursor-relative
APC peptide availability remains active and unachieved.
