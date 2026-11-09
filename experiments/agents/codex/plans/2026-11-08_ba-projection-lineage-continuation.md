# BA projection lineage and complete-development embargo continuation

Date: 2026-11-08
Owner: Codex / GPT-5
Status: projection verified; corrected immutable preflight continuation specified;
no GPU fit launched

## Trigger

The frozen primary preflight for experiment
`2026-11-08_1800_codex_unified-parity-retrain` passed the exact reference
manifest on 1,560,770 rows, then failed at byte zero of the first BA file:
`missing original measured-outcome lineage; prepare row dispositions first`.
The projection preserved physical archive lineage but omitted the canonical
`provenance.measurement_lineage` mapping required by `ObservationIndex` to
identify and reject duplicate measured outcomes across sources.

This is a projection defect. Do not weaken the common loader or synthesize a
protocol identity.

## Correction

1. Add a deterministic original-measurement identity to every projected BA
   observation:
   - namespace: the exact source archive SHA-256;
   - measurement ID: SHA-256 of the archive-native source key
     `(archive, peptide, identity, context)`;
   - replicate ID: empty because the archives expose no replicate identifier.
2. Retain the full physical member/fold/line-number lineage already recorded in
   `archive_lineage`; identical rows collapsed across folds remain one logical
   measurement.
3. Extend the development embargo from the affinity-only file to the complete
   exposed Stage B development union:
   - affinity development: exact peptide and shared 9-mer core;
   - presentation development: exact peptide, shared 8-mer core, or same-length
     Hamming distance one, matching the existing Stage B separation contract.
4. Deterministically transform the independently verified, hash-pinned v1
   admitted rows into a new immutable `prepared-v2` directory. This is equivalent
   to rerunning admission from the same archives because it only adds lineage or
   removes rows; it cannot recover, relabel, or add a v1-excluded source row.
   Keep the complete v1 physical-row disposition ledger as the parent audit and
   add a v2 supplemental disposition for every v1-admitted observation.
5. Never edit the completed v1 artifacts in place. Preserve v1 plus the failed
   common-loader receipt as audit evidence and record every v1-to-v2 hash.

## Verification and continuation

- Unit-test measurement-lineage stability across duplicate source folds,
  distinct archive namespaces, exact/core embargoes, and presentation
  core-8/Hamming-1 embargoes.
- Run and independently verify the corrected projection on Modal CPU with no
  model, outcome, prediction, TEST, locked cohort, network, or GPU access.
- Freeze a new retrain-preflight continuation contract pointing only to the v2
  BA hashes; do not mutate the already executed primary launch contract.
- Rerun both full canonical loader passes and independent verification. H100
  training remains prohibited until both corrected passes succeed.

## Frozen corrected-preflight implementation

Do not edit or reuse the executed v1 preflight output directory, call receipt,
launcher, or launch contract. Add a sibling `code/preflight_v2.py` and
`reproduce/v2/` bundle with new call names and output directories
`preflight-v2-primary` and `preflight-v2-verification`.

The only scientific manifest change from the failed preflight is replacement of
the two v1 BA sources by the independently verified v2 sources:

- MHC-I: 80,674 rows, 141,870,035 bytes, SHA-256
  `2c007e6b46a99835f95ec612aeb498f4e52940aa81c22809ad48b5e922d6d0ca`;
- MHC-II: 62,539 rows, 112,640,875 bytes, SHA-256
  `9fe752274c895cb4cd46e36dfeff43a3db73e0c96fa24ba495e1fcfd7f24c66a`.

Keep the original four Stage B sources, Stage A parent, model, optimizer,
balancing, selection, and no-TEST policy byte-for-byte identical. The effective
reference/parity manifests may differ only by the two v2 TRAIN sources.

For each CPU worker, stream-copy every pinned source and the parent checkpoint
once from the Modal volume into an isolated local scratch directory while
simultaneously checking byte count and SHA-256. Resolve both effective manifests
to those verified local copies. This changes transport only: it avoids repeated
network-filesystem traversal while the canonical `ObservationIndex` and
`training.modular_run.preflight` still parse every row independently for each
condition. Persist source-copy hashes/timings, complete condition receipts,
partition/census/model/training/scales/admission output, peak RSS, and proof of
zero TEST/model-prediction/optimizer/GPU access.

The primary worker must pass reference and parity before the verification worker
may start. The verification worker repeats the source copies and both complete
canonical passes in a fresh container, then requires exact equality for all
scientific outputs (rows, partitions, census, model, training, scales, and
admission) against the primary receipt. Runtime-only fields may differ. Any
failure stops before H100 launch and remains preserved as audit evidence.

## Primary-result transport continuation

The frozen primary call `fc-01M6YSQBZ1AKY6WW2NN1ZBQD9N` completed both
canonical passes and atomically committed a full receipt, SHA-256
`6efe6b3f64cbff080f9b03b9dce0ca1906a008d487fee3bd7d9eb8532ac6aaa2`,
before its return transport failed. The failure was not scientific: returning
the complete Python receipt exceeded Modal's inline-result threshold, which
attempted a blob upload from the intentionally network-blocked worker. Recover
the committed receipt directly from the volume; do not rerun the primary pass.

Freeze a transport-only source continuation for the independent verifier. Keep
all input hashes, scratch transport, canonical loader/preflight logic, comparison
keys, output path, blocked-network policy, and no-GPU contract unchanged. Change
only the function return value to a compact status plus receipt path/bytes/hash.
The local collector must download the complete committed verification receipt by
that artifact pointer and verify its byte count and SHA-256. Preserve the primary
call's remote transport exception and the recovered full receipt. The H100 gate
still requires the independent full receipt to pass exact comparison against the
recovered primary receipt.
