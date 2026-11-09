# NetMHC BA parity projection

Date: 2026-11-08
Agent/model: Codex / GPT-5
Status: complete; independently verified v2 TRAIN projection

Specification:
[`2026-11-08_netmhc-ba-parity-projection.md`](../agents/codex/plans/2026-11-08_netmhc-ba-parity-projection.md).

This family projects the already audited public NetMHCpan 4.2 and
NetMHCIIpan 4.3 BA training archives into source-faithful Presto TRAIN rows.
BA remains an `unspecified_assay/IC50` output with exact archive censoring.
The exposed current development peptide/core-9 neighborhoods are embargoed and
exact current-TRAIN pMHCs are deduplicated.

This is comparator-parity training data, not an independent benchmark. The
archive does not expose source studies, and no row becomes validation or TEST.
No model, prediction, current outcome, TEST, or locked confirmation cohort is
read. Large and redistribution-unresolved artifacts remain hash-pinned on the
`presto-checkpoints` Modal volume rather than Git.

The completed run used 8 CPU / 32 GiB / 512 GiB ephemeral disk, no GPU, and
blocked network access. A no-GPU contract is intentional because this is
deterministic data preparation rather than a performance-sensitive fit.

The first submission, Modal app `ap-E5gNDZmqElXSoi7Rw8Cp0T`, failed before
function creation because 128 GiB ephemeral disk is below Modal's current
512-GiB minimum for this request. No call ID was created and no input was read.
The retry changes only scratch capacity to the proven 512-GiB audit contract.

## Frozen execution

Source implementation commit `9043ef295c7bd1f4b47493ade18820366ceab483`
and launch-bundle commit `e7c9ca0d` pin the projection. Modal app
`ap-sKeyrUNZ5eFEpWFxrLcYec`, call `fc-01M6YNP3C4BYYBPACXC49NPSVD`, completed
in 58.12 seconds with 1,160,929,280 bytes peak RSS. Requested and observed GPU
are both null. Verification app `ap-skFXSxLmvLle9aU67Rw5uZ`, call
`fc-01M6YNS41PTJMGR2NFJV8SASVR`, passed every output hash and all 143,307 row
contracts.

## Lineage and complete-development continuation

The first frozen retrain preflight subsequently exposed a fail-closed defect:
v1 preserved physical archive lineage but did not map the archive-native source
key into canonical `measurement_lineage`, so the shared loader rejected the
first BA row before training. No GPU, optimizer, prediction, TEST, or locked
cohort was accessed. The corrective specification is
[`2026-11-08_ba-projection-lineage-continuation.md`](../agents/codex/plans/2026-11-08_ba-projection-lineage-continuation.md).

The immutable v2 continuation only annotates or removes independently verified
v1 admitted rows. It adds a measurement identity namespaced by source-archive
SHA-256 and extends the embargo to the full exposed Stage B development union:
presentation exact peptide, shared 8-mer core, and same-length Hamming distance
one, in addition to the inherited affinity exact/core-9 embargo. It cannot add
or relabel an excluded source row.

Source commit `477196db03967443bc8e0ff7f7a8e90453d97613` and launch-contract
commit `493ceea9` pin this continuation. Projection app/call
`ap-iLZ81MzNcVpTs4U2aEnfok` / `fc-01M6YS3HZ79409AHMV4RVR2G1H` completed in
55.34 seconds with 3,895,259,136 bytes peak RSS. Verification app/call
`ap-mQLkR0DdAnawlxMiat9ooB` / `fc-01M6YS6XKN48FXAWAH00SD6X9Z` independently
re-read all outputs and passed.

Of 143,307 v1 admitted rows, v2 retains 143,213: 80,674 MHC-I and 62,539
MHC-II. It removes 91 shared-core-8, one exact-peptide, and two Hamming-1
presentation-development neighbors. Every retained row has a unique canonical
measurement ID and observation ID. Requested/observed GPU remain null, network
is blocked in the workers, and model/prediction/TEST/locked access is false.

The projection scanned 343,914 physical BA rows. Deterministic dispositions
were:

| Disposition | Rows |
|---|---:|
| admitted TRAIN | 143,307 |
| exact current-TRAIN pMHC dedup | 138,168 |
| nonhuman/unresolved MHC | 43,414 |
| exposed-development exact peptide embargo | 11,552 |
| exposed-development core-9 embargo | 7,451 |
| conflicting eligible source targets | 20 |
| noncanonical peptide | 2 |

The source inventory contains 640 conflicting nonnumeric keys and 137 exact
duplicate keys in total. Disposition precedence first removes nonhuman or
unresolved rows, so most source conflicts do not reach the eligible-human
conflict disposition. No conflict or duplicate is admitted.

The admitted cohort contains 80,744 MHC-I and 62,563 MHC-II rows across 108 and
60 archive identities. It preserves 121,447 exact interior values, 19,547
`>=50000 nM` bounds, and 2,313 `<=1 nM` bounds. Every row maps to the reference
IC50 decoder with unknown preparation, mechanism, and detector; no archive row
is presented as support for a protocol-specific assay output.

## Artifacts

- `netmhcpan42-ba-train.jsonl`: 114,297,607 bytes, SHA-256
  `4491918763c1cf1c64825af0228357f43516293d1b6ea73ef8a6a7554d4635df`.
- `netmhciipan43-ba-train.jsonl`: 91,225,030 bytes, SHA-256
  `153a51f8997093c0b8cf724dc8ff649a676ee3ee2e516c52dff810bb18d00ee2`.
- `source-row-dispositions.jsonl.gz`: 29,476,224 bytes, SHA-256
  `4d35b9008d2236bc0c0ffd8d9e2c951a273ac2eb576b2e848f83a584521cbe59`.
- Conflict and duplicate ledgers are separately hash-pinned in
  `results/summary.json`.

All paths are under
`modal-volume://presto-checkpoints/2026-11-08_1740_codex_netmhc-ba-parity-projection/prepared/`.
The 205.5-MB row outputs and 29.5-MB disposition ledger remain off Git because
the source archive's redistribution status is unresolved. The small summary,
projection/verification receipts, calls, scripts, and hashes are committed.

The training continuation must use `prepared-v2`, not v1:

- `netmhcpan42-ba-train.jsonl`: 141,870,035 bytes, SHA-256
  `2c007e6b46a99835f95ec612aeb498f4e52940aa81c22809ad48b5e922d6d0ca`.
- `netmhciipan43-ba-train.jsonl`: 112,640,875 bytes, SHA-256
  `9fe752274c895cb4cd46e36dfeff649a676ee3ee2e516c52dff810bb18d00ee2`.
- supplemental v1-admitted-row dispositions: 5,581,707 bytes, SHA-256
  `c9c7a384e6affe555d2fae3a6e8ba37c324763bc586c43c7b5a84107d7ad9f3d`.
- v2 summary: SHA-256
  `ec21c6e9aae67c71345a256c0b62231339d73e2f2dffe6942863314c0c20c08b`;
  independent verification: SHA-256
  `f41a64dc6a6bb94bca78044b45d166f0f86641b90a589fe86fedd054f529346c`.

These large artifacts live under the same Modal-volume family at
`prepared-v2/`; only the small receipts are committed.

Validation/test metrics and prediction dumps do not apply: this family creates
TRAIN data only and deliberately loads no model.

## Decision

Admit both verified **v2** files to the data-only retraining arm after the canonical
combined-manifest loader/preflight passes. Treat this strictly as comparator-
parity training. It cannot support a NetMHC superiority claim and does not
replace the planned genuinely new study-disjoint affinity TEST.
