# MHC-II MS prediction with length-matched backgrounds

Status: native101 recovered and all logistic controls independently verified;
native102/permuted101 continuation active. Full comparison remains incomplete.
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

Verified logistic controls use binding-only,61sequence features, and their
62feature combination. Standardization and coefficients use DP training only.
Objective: meanBCE +.01squared slope norm, unpenalized intercept; float64
L-BFGS-B,300iteration cap/ftol1e-12/gtol1e-8; fail on nonconvergence.

Report every arm's validation/DP test/DQ8 test BCE, Brier, threshold.5
classification metrics, AUROC/AUPRC and per-example predictions. IC50 nM/500nM
metrics do not apply to these labels. Binding retention is checked by exact
parameter/scaler comparison and fresh no-grad probes to2e-5log10(nM).
The plan freezes2% BCE improvements against all three controls on both test
sets, AUROC retention, separation from the permuted diagnostic and2,000 paired
component-bootstrap draws. First-seed reported metrics are below; no complete-comparison success is claimed.

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

## 2026-10-27 recovery and available outcomes

Original native101 finished3epochs/2,592updates, then exceeded4GiB RSS while
serializing its training evaluation cache (2,092.293s,4,315,611,136bytes peakRSS).
No final validation/test cache evaluation had started. Original failed receipts,
logs and all checkpoint bytes are preserved. A separate checkout at2ed1bd8 changes
only JSON persistence/hashing in `training/modular_resume.py`, streaming identical
bytes/hashes with unchanged model mathematics and strict resume contracts.

The memory regression failed before the fix and passed afterward. Focused suite:
50passed; full suite:2,666passed,4skipped,33warnings in609.05s. Byte/digest identity,
atomic failure preservation and bounded serializer allocations are covered.
Only `atomic_json`/`digest` ASTs differ; old/new runtime maps and source snapshots
are frozen in `reproduce/serialization-transition.json` and `serialization-fix/`.

Evaluation-only recovery completed in213.045s/2,188,427,264bytesRSS, with zero new
optimizer updates. Selected/final tensors agree; all32retained IC50 dependency
tensors and scaler match the parent exactly, and20fresh probes have zero error.
Parameter/exposed-feature coverage passes. All four per-example prediction dumps
are retained. These first-seed metrics are reported by the canonical evaluator;
the independent all-arm replay, ensemble and bootstrap remain pending.

| Fixed native101 partition | n | BCE | AUROC | AUPRC | Balanced accuracy | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DP train |55296|.555176|.791221|.780887|.714753|.764612|.728300|
| DP validation |11040|.567887|.774720|.773374|.699728|.739855|.711312|
| DP test |11808|.567756|.775586|.772586|.701558|.748645|.714979|
| DQ8 external |6906|.699868|.633184|.649963|.593976|.448016|.524585|

The first seed has weaker external discrimination and external BCE above the
constant0.5 reference's log(2). This does not establish a native-model advantage,
calibration, processing identification or a completed MS qualification. Keep the
frozen remaining conditions and all prospective success gates.

Native102's first attempt stopped before initial checkpoint/training on the10GiB
available-RAM floor:19.736s/290,471,936bytesRSS. Admission/config and the empty
lock are hash-preserved under raw `native102-preflight-attempt01`; no state was
overwritten or falsely resumed. A new explicit controller requires20GiB available
RAM continuously for60seconds before each action, with a30minute admission
deadline. Original action guards remain unchanged; failed actions never retry
automatically. Native101 evaluation is not repeated.

`remaining-arms-plan.md`, `remaining-launch.json`/`remaining.sh` freeze this state
chain. `comparison-plan.md`, `comparison-launch.json`/`comparison.sh` and source
snapshots freeze independent source/metric/coefficient replay, the fixed ensemble
and2,000component bootstrap. Nine artificial reducer tests pass; the initial
missing code-directory import-path failure is preserved. Live status belongs to
`reproduce/remaining-controller-state.json`; publication uses a separate snapshot.

The original canonical fit remains failed; recovered evaluation is explicitly
separate. Full comparison, binding refinement and fitted pathway/condition
qualification are still required. No serving or baseline promotion.

## 2026-10-27 verified comparator closure

The additional all-arm admission controller reached its 1,802.965-second deadline
without launching any action. Its failed state is preserved. A separately frozen
comparator-only schedule then ran the already planned independent controls before
the remaining native fits: original 13 GiB startup/10 GiB runtime RAM limits,
4 GiB RSS/2 h action caps, two numerical threads/one interop. The comparator schedule
required 13 GiB continuously for 30 seconds with a 15-minute admission deadline;
this explicitly differs from the failed extra 20 GiB all-arm scheduling gate.
No source, feature, optimizer, split or predictive gate changed.

Feature extraction took 190.526 s / 1,644,609,536 bytes peak RSS; all three logistic
fits took 6.380 s / 950,861,824 bytes. Independent verification took 15.081 s /
1,315,536,896 bytes using one thread. It replays all 85,050 original source rows,
255,150 control predictions and 80 fresh parent probes. An independently implemented
Newton solver agrees with the frozen fitted convex objectives to 5.44e-15 and
coefficients to 6.58e-8. Three artificial derivative/solver tests pass. Original
saved predictions, not the verification refit, determine the reported metrics.
Native101 metrics are independently reduced from its saved predictions; full
native inference replay remains part of the final all-arm verification.

| Condition | DP validation BCE / AUROC | DP test BCE / AUROC | DQ8 external BCE / AUROC |
|---|---:|---:|---:|
| Binding-only logistic | .671782 / .617169 | .674997 / .606370 | .661982 / .726061 |
| Sequence logistic | .647567 / .670653 | .654715 / .656213 | .679688 / .609729 |
| Combined logistic | .624406 / .708389 | .630023 / .700584 | .593876 / .794303 |
| Native101, single-seed diagnostic | .567887 / .774720 | .567756 / .775586 | .699868 / .633184 |

The single native seed improves DP-test BCE by 9.88% relative to the combined
control, but worsens DQ8-external BCE by 17.85%. This does not establish the
prespecified native ensemble's success or failure: native102, permutation,
ensemble and paired bootstrap are still required. The combined control is the
strongest of the three controls on both test sets; it is the current transfer
reference, not a substitute for completing the planned comparison.

All four partitions' loss, Brier, accuracy, balanced accuracy, precision, recall,
F1, AUROC and AUPRC are in `results/logistic-controls/verified-point-metrics.csv`
and `independent-verification.json`. Full source-linked prediction dumps, training
features, standardization and coefficients are retained in that directory.
`comparator-only-launch.json`/`.sh`, `comparator-verification-launch.json`,
`verify-comparators.sh` and source snapshots freeze the executed recipes.

`native-remainder-plan.md` and `native-remainder-launch.json`/`.sh` now continue
only native102 and permuted101. Their action/admission loop is AST-identical to
the previous controller: 20 GiB for 60 seconds, at most 30 minutes before each
action, original action floors and no retries. Comparator evidence and original
recovery/failure receipts are hash-pinned. Both intended fit directories were
absent at launch. Its separate state is `native-remainder-controller-state.json`;
the failed older state remains failed. No new censor fit overlaps it.

The DQ8 cohort is now inspected development evidence. Any later model revision
requires a new prospective recipe and a fresh qualification cohort. No isolated
processing, condition-effect, natural prevalence or calibration claim follows.
