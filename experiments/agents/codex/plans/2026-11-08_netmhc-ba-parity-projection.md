# NetMHC BA parity projection and class-II decision gate

Date: 2026-11-08
Owner: Codex / GPT-5
Status: specified before implementation

## Objective

Project the already audited public NetMHCpan 4.2 and NetMHCIIpan 4.3 BA
training archives into source-faithful Presto TRAIN observations. This is a
comparator-parity data arm, not an independent evaluation cohort. Preserve an
explicit unspecified-assay IC50 output, censoring, archive lineage, exact MHC
grooves, and every admission/exclusion decision.

The projection must be sufficient for a data-only affinity retrain while
keeping the exposed 1720 development peptide neighborhoods embargoed. It must
not access a model, prediction, TEST, locked confirmation cohort, or outcome
from the current development rows.

## Frozen inputs

- NetMHCpan 4.2 training archive, 269,075,958 bytes, SHA-256
  `243a3075887fef3c975da232a87e8d1e17b1aafbf7dba4e52180361d657b5d14`.
- NetMHCIIpan 4.3 training archive, 139,440,263 bytes, SHA-256
  `b3602edd48fd34a2d4dcd574bea7781bb75c50f9f4dbc3a2d7178301b735c521`.
- Completed semantic-admission summary, SHA-256
  `b4f86522dd5fb0a22a90261d979613dec03d17c229f4a3a146410f563b9cd91f`.
- Admission decisions, SHA-256
  `84fb6dfeec2e6c64f227721d7d99236e2079038f8fc9efc56d6b3e451c499d0a`.
- Frozen human MHC catalog, 61,469,362 bytes, SHA-256
  `0e26d38ed0c572692a3486e100272c56bd1134ce74d2b6122fa4be2f4c6b0a82`.
- Current TRAIN identity file, SHA-256
  `acc22d1481ce8e0b45a6df86ca9a62e70721752955070b7eb26b85e48361635d`.
- Exposed development identity file, SHA-256
  `08e41a6c006274b5929142fc6bd4687a289a08df9cc5212edd92399f907f2f8a`.

All large and license-unresolved inputs remain on the `presto-checkpoints`
Modal volume. The Git experiment bundle records only hashes, summaries,
scripts, and small receipts.

## Source semantics and target mapping

1. Read only the five `c00[0-4]_ba` members from each frozen archive.
2. Admit only canonical peptides and identities resolving to complete human
   class-I or paired class-II groove sequences through the frozen catalog.
3. Treat BA as an archive-native `unspecified_assay/IC50` measurement. Do not
   assign purified/cellular, competitive/direct, or detector labels that the
   archive does not provide.
4. Invert `y` as `IC50_nM = 50000 ** (1-y)`.
   - `y == 0`: store `50000 nM` with qualifier `>=`.
   - `y == 1`: store `1 nM` with qualifier `<=`.
   - `0 < y < 1`: store an exact `=` value.
5. Preserve source archive, archive hash, member, fold, literal identity,
   archive context, original transformed target, resolved catalog alleles, and
   groove sequences in provenance.
6. Keep the source study explicitly unknown. Archive folds are not studies and
   must not be represented as publications or donor groups.

## Deterministic admission and embargo order

Apply the following label-independent disposition order and record counts:

1. malformed peptide/column/target;
2. nonhuman, unresolved, or incomplete groove identity;
3. source pMHC key with conflicting numeric targets anywhere in the archive;
4. duplicate exact source row, retaining one canonical representative and all
   contributing fold/member lineage;
5. any exact peptide or peptide sharing a 9-mer core with the exposed current
   development set, irrespective of MHC identity;
6. exact peptide-plus-groove pMHC already present in current TRAIN;
7. admitted TRAIN row.

No archive row becomes validation or TEST. No rule may use a current outcome,
model score, comparator score, or downstream metric. Current TRAIN is read only
for pMHC identity deduplication; development is read only for peptide/core-9
embargo.

## Outputs

Write immutably under one new timestamped experiment path on the Modal volume:

- class-I and class-II Presto observation JSONL files;
- deterministic gzip disposition ledger for every recognized source row;
- source-key conflict and duplicate lineage tables;
- summary JSON with class, allele, length, qualifier, fold, exclusion, overlap,
  and output counts;
- artifact receipt with bytes and SHA-256 for every output;
- runtime/resource receipt and exact input hashes.

Every admitted row must parse through `Observation.from_dict`, preserve schema
`presto.modular.observation.v1`, use split `train`, and have a deterministic
observation ID. The output assay descriptor remains unknown on every method
axis so the shared factorized decoder learns the reference IC50 output without
forging protocol-specific support.

## Verification

1. Unit-test target inversion, boundaries, source-key conflict exclusion,
   exact duplicate collapse, core-9 development embargo, TRAIN pMHC dedup,
   deterministic IDs, and output parsing on synthetic fixtures.
2. Run the frozen projection on Modal CPU with network blocked. GPU is
   intentionally `null`; this is data preparation, not a performance run.
3. Independently collect and verify all output hashes and invariants.
4. Register the completed family in `experiments/experiment_log.md`; state
   explicitly that it has no validation/test predictions because it loads no
   model and creates TRAIN data only.
5. Before training, run the canonical common-loader preflight on the projected
   files and freeze the combined current-plus-parity manifest.

## Class-II model decision gate

The current shared-v7 model already enumerates all class-II 9-mer registers,
scores every candidate, and performs normalized log-sum-exp marginalization.
The prior temperature/sharpening audit was negative. Therefore the mechanism
arm will not add another nominal register-pooling layer.

Run the data-only arm first. If class-II development remains materially below
NetMHCIIpan after parity-data training, preregister one compact architecture
arm that changes an actual missing interaction:

- attend each candidate core separately to the alpha groove and beta groove;
- fuse the two chain-aware interaction vectors through a zero/neutral-start
  residual adapter into the existing shared candidate representation;
- retain the same register enumeration, normalized marginalization, shared
  encoder, assay decoder, training budget, and data;
- report register entropy/core diagnostics and per-study/per-assay gains;
- reject the adapter unless it improves new development with no class-I or
  existing-task regression.

This is an ablation, not a default architecture change. Independent superiority
still requires genuinely new study-disjoint affinity TEST data that neither
archive can provide.
