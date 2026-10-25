# Shared intact-degradomics source: canonical admission and independent replay

Codex / GPT-6, October 25, 2026. Registered before canonical materialization.
Source audit: [2338](../2026-10-24_2338_codex_intact-degradomics-source/README.md).
Implementation: [Hitlist831](https://github.com/pirl-unc/hitlist/issues/831).

Status: canonical source build and independent replay pass; no predictive result. This family
builds a standalone shared Hitlist source package and independently compares its
public query output with original source cells, masses, assignments, quantities,
acquisition identities and publisher reconciliation. Producer replay is an
additional software integrity check, not the independent verifier.

The exact 14 original assets are pinned in the copied source contract. All eight
workbooks are retained, including short fragments, unassigned masses, ambiguous
assignments, errors/missing values, repeated views and the publisher disagreement.
No training selection, split, imputation, negatives, labels, rates or model is
created here. No validation/test predictions or binding metrics apply because
source admission precedes a separately registered kinetic experiment.

Read `reproduce/plan.md` and `reproduce/launch.json` for source and resource
contracts. Use `reproduce/launch.sh build` then `reproduce/launch.sh audit`.
The launch refuses prior attempts and freezes each executed script. All training
and pretraining parameters, losses and output mappings are inapplicable.

Local CPU, one numerical thread; no Modal, GPU, installs or shared-runtime changes.
Resource limits: 20 minutes/stage, 3 GiB RSS, 128 MiB additional physical storage,
1 GiB free-space floor plus 32 MiB startup margin. Original and failed artifacts
are preserved. Raw data live under the source family; the canonical package lives
under `data/intact_degradomics_canonical/` with the exact path in launch.json.

Biological scope: one human substrate with mouse-organ-derived purified 20S,
reported from one animal. Nominal zero contains products. Reported replicate
labels do not establish independent reactions/specimens. MS intensities are not
calibrated molarity. No APC concentration, new-protein or biological predictive
validation is established by a successful source replay.

## Verified results

The canonical package preserves 14 assets, 133 source views, 102 acquisition
identities, 410,349 explicit XML cells (325,718 populated), 2,991 formula/cache
records, 17,862 input-mass rows, 8,138 assignments and 44,457 quantities. All source
cells and quantities independently reconcile; original openpyxl search-row
interpretation checks every input mass and assignment. Public joined queries
preserve all 8,138 assignments and their quantities/acquisition/missing-ID scope.
All 3,427 publisher identity links reproduce the earlier independent source map;
one quantity disagreement and nine repeated acquisition views remain explicit.

| Stage | Wall seconds | Peak polled RSS | Outcome |
|---|---:|---:|---|
| Build including producer replay | 13.37 | 659.4 MiB | Pass |
| Initial independent audit | 12.14 | 943.2 MiB | Verifier omitted substrate-only label grammar; failed attempt retained |
| Amended independent audit | 12.25 | 946.7 MiB | Pass; all original invariants retained |

The [control-label amendment](reproduce/control-label-amendment.md) fixes verifier
coverage without changing the canonical package. All stages stayed within caps.
Exact script hashes, resource measurements, source manifest and per-workbook
counts are in `results/`. Format/lint and 45 relevant Hitlist tests pass. Full
local suite remains deferred below its 2 GiB free-space gate; hosted full checks
and shared-code release are still required.

Decision: source evidence is sufficiently faithful for a separately specified
kinetic-candidate contract. This does not certify all assignments or resolve
unknown MS response, initial product pool, material independence or molarity.
No model, supervised cohort or predictive score was produced.

## Kinetic support and next experiment

The predeclared descriptive policy retains1,278of2,617F2assignment cells,
237sequences. Nonexclusive reasons are1,328ambiguous assignments,10cells in
multiple-mass groups and3input-link/intensity conflicts. Intact substrate is
available in all21acquisitions;80sequences have at least two early times and one
late time. Support analysis took8.12s/748.5MiB and did not fit or score a model.
All row dispositions and time/sequence support are saved in `results/`.

This permits a prospective product/substrate-ratio time forecast. Shared run
scaling can cancel within acquisitions; fixed peptide-specific response still
requires training-time anchoring. The next [model specification](../agents/codex/plans/intact-kinetic-forecast.md)
compares direct temporal controls with a minimal mass-conserving fragmentation
model and matched conditional score gradients. The latter is explicitly a hybrid
with exact forward probabilities, not simulator-only RL. No future held-out
performance or APC concentration validation is claimed by this support census.

Shared source implementation is [draft Hitlist832](https://github.com/pirl-unc/hitlist/pull/832),
commit7ece0885b8423f4aae205bfdf36b801d2106c842; all frozen runtime files match.
Hosted full checks, merge and source release remain pending. Hitlist831 remains
open for unresolved source metadata and subsequent kinetic qualification.
