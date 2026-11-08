# NetMHC BA parity projection

Date: 2026-11-08
Agent/model: Codex / GPT-5
Status: implementation frozen; execution pending

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

The run will use 8 CPU / 32 GiB, no GPU, network blocked. A no-GPU contract is
intentional because this is deterministic data preparation rather than a
performance-sensitive fit.

## Planned artifacts

- `prepared/netmhcpan42-ba-train.jsonl`
- `prepared/netmhciipan43-ba-train.jsonl`
- `prepared/source-row-dispositions.jsonl.gz`
- `prepared/source-target-conflicts.jsonl.gz`
- `prepared/source-exact-duplicates.jsonl.gz`
- `prepared/summary.json`
- independent remote verification receipt

Validation/test metrics and prediction dumps do not apply: this family creates
TRAIN data only and deliberately loads no model.
