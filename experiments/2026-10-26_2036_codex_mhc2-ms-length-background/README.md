# MHC-II MS prediction with length-matched backgrounds

Status: cohort independently verified; first native fit active, comparators queued.
Codex / GPT-6,2026-10-26. Registered at
`aa5eb5ed41e2916b8f94c9e4a30e9d069bf3a5fc` with recorded experiment/task dirt.
[Full amended plan](../agents/codex/plans/2026-10-26_mhc2-ms-length-background.md)
inherits the original predictive plan's model arms, gates and limitations.

The previous exact-composition background failed for a homopolymer. This separately
registered family retains every original positive and partition, and generates
length-matched controls from DP training-positive amino-acid frequencies. Controls
are explicitly synthetic; this measures detection enrichment against that
distribution, without measured nondetection or isolated processing claims.

## Source, background and partition contract

Qualified human class-II MS sources: DP/K562 PMID35251023 and external DQ8
PMID34433824, from the independently audited1954 family. Only admitted positive,
source-unmodified observations with one explicit complete paired molecule enter.
DQ8's source includes spliced-peptide analyses; linear origin is not asserted.
The native export remains Hitlist1.78.0, SHA256
`f902341e90c1c57defc93f4bc540ffcbea1b41061afb4492ddd8aaa7642c65bd`;
admitted observations SHA256
`532977c4a5e9b6530fca36d039b73a3b832c7b15197dc7a6def0bd66191aecef`.
Current curator notes are separately pinned2.0.42 evidence.

Collapse identical study/peptide/groove-pair units, retaining all original IDs and
byte offsets. DP components connect any shared contiguous nine-residue segment;
fixed SHA256 assignment gives70/15/15 target proportions. Remove evaluation
components exposed to any original IC50 input partition. DQ8 excludes segments
shared with any DP peptide or any original IC50 input. Donor, homology, parent
and search independence are not established. No prediction determines membership.

| Partition | Distinct positive peptides | Components | Positive/control examples |
|---|---:|---:|---:|
| DP train |18,906|5,986|27,648 /27,648|
| DP validation |3,853|1,303|5,520 /5,520|
| DP test |4,062|1,317|5,904 /5,904|
| DQ8 external test |3,453|1,189|3,453 /3,453|

The30,274 generated sequences use one length-matched draw per positive peptide,
reused across that peptide's molecule units. Frequencies use distinct DP training
positives only. Reject original/known-positive sequences (1,106,409 in the pinned
full native MS source), duplicate controls, cross-partition segment collisions and
evaluation initializer overlap. All42,525 positive units remain;43,952 original
source rows are retained in the lineage table. Context inputs stay unasserted;
original condition metadata remains available through the immutable source links.
No causal cell/pathway effect is trained from this study contrast.

## Fixed training and evaluation

Common parent: original seed42 IC50 selected checkpoint, SHA256
`db2585e63489cb5aac8c7a32fd7e522bcbc51035846c05f8cd2794c4449e3dc5`.
Its entire IC50 parameter dependency set and inherited scaler are frozen. New
`elution_observation` parameters train under canonical `fit_index`. MS labels1/0
map to total elution logits with BCE; cellular/detector component scores remain
unidentified individually. No identity loss, new quantitative label or censoring.

Native seeds101/102 and one paired training-label permutation diagnostic use
three complete epochs, batch64, exactly2,592 updates, AdamW3e-4/weight decay1e-4,
dropout.1, checkpoint every50steps. Primary predictions use fixed-final checkpoints;
the two true-label logits form the prespecified ensemble. Canonical selected
outputs are secondary. No seed or checkpoint is chosen from test outcomes.

Queued logistic controls use binding-only,61sequence features, and their
62feature combination. Standardization and coefficients use DP training only.
Objective: meanBCE +.01squared slope norm, unpenalized intercept; float64
L-BFGS-B,300iteration cap/ftol1e-12/gtol1e-8; fail on nonconvergence.

Report every arm's validation/DP test/DQ8 test BCE, Brier, threshold.5
classification metrics, AUROC/AUPRC and per-example predictions. IC50 nM/500nM
metrics do not apply to these labels. Binding retention is checked by exact
parameter/scaler comparison and fresh no-grad probes to2e-5log10(nM).
The plan freezes2% BCE improvements against all three controls on both test
sets, AUROC retention, separation from the permuted diagnostic and2,000 paired
component-bootstrap draws. No predictive metric or success claim is available yet.

## Verification, artifacts and runtime

Preparation14.509s/874,315,776bytesRSS; independent verification14.810s/
858,882,048bytesRSS. The latter rereads original rows/byte offsets, rebuilds
components by graph flood fill, replays controls using inverse-CDF draws, and
checks every source/control observation. All85,050examples pass. Ten software
boundary tests pass, including unchanged partition logic, training-only frequency
construction, homopolymer retention and positive/exposure collision rejection.

LocalCPU only; no Modal/GPU/package changes. Preparation/replay use one numerical
thread. Native fitting uses two threads/one interop,4GiB RSS/2h action cap,13GiB
startup RAM and10GiB runtime RAM floor,15GiB startup/10GiB runtime free disk.
No native MS fit overlaps censor recovery. Controllers stop without retry.

Configuration deviation: native-manifest preflight used the fit wrapper's two
threads and4GiB/2h guard instead of the plan's one-thread2GiB/300s preparation
guard. Actual use was55.881s/307,363,840bytes, below the stricter time/RSS bounds;
no predictive scoring occurred. This does not change the subsequent fit contract.

`reproduce/launch.sh`, `launch.json`, `verify-cohort.sh`, `native.sh`, `native-launch.json`,
`logistic.sh`, `logistic-launch.json` and frozen sources pin all invocations.
`results/` contains source lineage, dispositions, frequency/control generation,
cohort summaries and independent verification.201,259,710bytes of prepared source
JSONL remain under `/private/tmp/presto-mhc2-20261026/ms-length-background/cohort/`;
exact file hashes are in `results/cohort.json`. Raw native fits/null-source copies
remain under the adjacent `native/` directory. No completed eval is claimed;
validation/test predictions and full comparison closure are pending.

**Next:** follow all native/control jobs, independently verify scores/metrics and
retained IC50 behavior, close plots/README/canonical log. The broader goal remains
active; distinct processing-pathway/condition learning is still unqualified.
