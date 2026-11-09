# Length-matched MS background after infeasible permutation controls

Codex / GPT-6, 2026-10-26. New family:
`experiments/2026-10-26_2036_codex_mhc2-ms-length-background/`.
This is a new generated-background contract, prospectively registered before any
model scoring, fitting or held-out prediction access. The2030 family is terminal
failed preparation, not an unsuccessful predictive result.

The [original detailed plan](2026-10-26_mhc2-ms-prediction.md) remains in force
for source qualification, representative units/lineage, source/initializer
exclusions, deterministic component partitions, model/control arms, all training
parameters, metrics/gates/bootstrap, frozen IC50 preservation and resource limits.
The only scientific changes are the background distribution and the explicitly
corresponding scope of interpretation. Raw outputs now belong under
`/private/tmp/presto-mhc2-20261026/ms-length-background/`.

## Failure and revised control distribution

The original prepare action stopped after8.395seconds, peakRSS863,846,400bytes.
A length15 homopolymer has one possible sequence at fixed composition, so it
cannot yield a distinct permutation. No model was scored or fitted. All source
positives, original scripts/receipt, the failed control's hash and partition
counts remain preserved. Do not drop that example or quietly relax the old rule.

For the new family, derive the20-residue frequency vector from the concatenated
**distinct DP training positive peptides only**, after frozen partition/exposure
rules. Weight each residue occurrence once per distinct peptide; no validation,
test, DQ8, model score or binding outcome affects this distribution. Require all
20standard residues to have positive support; otherwise stop.

For each retained positive peptide, use a local PRNG seeded from SHA256 of
`presto-ms-length-background-v1|` plus its sequence. Draw each residue independently
from the fixed training frequency vector, preserving peptide length. One control
is reused for all molecule units of the parent peptide. Try at most1000 draws.
Retain every original collision rejection: unchanged/known-positive sequence,
previous generated sequence, cross-partition nine-residue overlap, and
initializer overlap for evaluation controls. Keep sorted partition/peptide order.
Fail if a control cannot be drawn; no automatic retry or dropping positives.
Preserve frequency counts/vector, seed/attempt/parent IDs and actual compositions.

Controls are now **length-matched, training-composition-distributed**, not
per-peptide composition matched. The original sequence and combined controls
include full composition features and remain mandatory. These tests estimate
detection enrichment against this explicitly artificial distribution; they do
not measure biological nondetection, natural prevalence or isolated processing.
The original fixed gains/gates remain unchanged and cannot be loosened to get
a positive result.

## Implementation and verification

Copy the frozen2030 preparation/helper sources into the new family before edits.
Preserve shared component/partition logic exactly; change background drawing,
metadata and paths. Verify unchanged helper behavior and new boundary invariants:
training-only frequencies, length preservation, deterministic controls, known
positive/exposure/cross-partition rejection, and a homopolymer positive receiving
a distinct possible control. Do not execute unchanged old permutation tests as if
they tested the new distribution. Independent cohort replay remains mandatory
before any predictive run.

The first preparation established DP train18,906/validation3,853/test4,062 and
DQ8 external3,453 distinct retained peptides, with5,986/1,303/1,317/1,189 connected
components respectively. Those counts were obtained without prediction outcomes
and must remain exactly the same under the revised background-only contract.

- [x] Prepare revised controls without changing positive membership/partitions.
- [x] Independently verify source lineage, boundaries and generation.
- [ ] Freeze model manifests; run all specified controls/native arms.
- [ ] Close complete validation/test outcomes and retain IC50 behavior.
