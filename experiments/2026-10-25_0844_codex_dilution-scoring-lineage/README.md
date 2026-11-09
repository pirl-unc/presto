# PXD024809 scored/aligned and query-library lineage — completed source audit

The deposited score tables explain 49,646 of 64,070 aligned slots by exact
feature-ID matches. Their intensities are numerically identical. The remaining
slots retain distinct missing-ID and absent-from-export states. All nine observed
heavy standards belong to the target library; the tenth intended standard is
absent from the library, score exports and aligned table. It is not a negative
measurement. These findings support shared source ingestion, not a calibrated
APC availability model or an unconditional MS detection likelihood.

## Question, registration and source contract

Agent/model: Codex / GPT-6; October 25, 2026. Base commit
`e232719fa0d0cf7e574bb2d558ebcac74edec0e1`, dirty with unrelated task/history
artifacts preserved. This follows [0813](../2026-10-25_0813_codex_dilution-observer-admission/README.md)
and [Hitlist #845](https://github.com/pirl-unc/hitlist/issues/845), after the native
response-only search triggered its stopping rule in family0747.

The [plan](reproduce/plan.md) selected all ten public per-run `.mzML.tsv` score
exports and `Dilute_library_merged_decoy.tsv` before their full contents were
inspected: 11 assets, 389,468,698 bytes. Retrieval hashes are newly computed SHA256s;
the PRIDE inventory supplied no deposit checksums. Reuse the prior aligned TSV,
protocol/standard panel and pinned pipeline source by hash. No RAW/mzML search.

Original new assets remain once under
`data/dilution_observer_sources/2026-10-25_0844_codex_dilution-scoring-lineage/`;
0813 inputs remain in its sibling directory. Per-stage configs pin paths, hashes,
URLs, limits, git/environment and the exact executed code. No score filter,
imputation, censor substitution, form collapse or row selection is applied.

The complete source universe is 1,031,853 rows / 29,039,248 string cells across
12 tables: 76,561 scored candidates, 891,222 library transitions and 64,070
aligned slots. The library has 12,851 group/form/charge/decoy memberships. Feature,
run and group IDs remain text, including identifiers larger than 2^53. There are
no duplicate scored `(filename,id)`, library transition IDs or aligned
`(run,precursor)` keys. Asset/run name links use only the registered `#` to `_`
normalization and do not establish physical specimen identity.

## Results and interpretation

| Aligned-to-scored disposition | Slots | Intensity / score state |
|---|---:|---|
| Exact feature/run ID; form, charge, precursor and target agree | 49,646 | Positive intensity and score; intensity exactly equal |
| Feature ID absent from deposited scored export | 3,485 | Positive intensity and score; reason unresolved |
| Missing feature ID | 4,932 | Positive intensity, missing score |
| Missing feature ID | 5,743 | Literal zero intensity, missing score |
| Missing feature ID | 264 | Literal NA intensity and score |

All 49,646 matched intensities, RTs and peak ranks are decimal-equal. Integration
boundaries and m_scores are equal or within the fixed
`abs(delta) <= 1e-12 + 1e-12 * max(abs(a),abs(b))` comparison tolerance. Raw values,
signed deltas and comparison labels are preserved; tolerance is not proof of
common rounding or an identification acceptance rule. Every aligned slot appears
in [the compressed comparison dump](results/aligned-feature-links.tsv.gz).

The initial library-number hypothesis failed: only 20 aligned slots and 20 scored
rows agree when library `TransitionGroupId` is equated to scored/aligned precursor
numbers. Both namespaces and the failed audit remain intact. The prospectively
registered [amendment](reproduce/membership-contract.md) uses exact full molecular
form, charge and target/decoy membership, without equating the IDs. It matches all
76,561 scored rows and all 64,070 aligned forms unambiguously. Additional literal
group-label agreement holds for 64,060 aligned rows; the other ten are the same
form/charge with empty aligned label versus literal `NA` in the library. They
remain missing-label differences, not a different molecule or an inferred label.
Both source namespaces are retained in [the compressed mapping](results/namespace-mapping.json.gz).

The original ten-standard protocol panel is retained. Nine exact heavy forms
each have 84 target transitions, no decoy-form matches and ten aligned slots.
There are 195 scored standard candidates (multiple peak candidates can exist).
Of the 90 aligned standard slots, 87 have exact feature matches. All three
unmatched standard slots have missing IDs/scores: one positive intensity and two
zeros for the same standard. Do not convert them to measured concentrations,
known censor bounds or independently scored signals. The tenth intended standard
has no target or decoy form, even under bare-sequence membership. Full positions,
forms, counts and locators are in [standard memberships](results/standard-memberships.json)
and [standard links](results/standard-links.json). The latter deliberately retains
the original failed numeric-library-ID check; use the amended mapping for actual
form membership.

Pinned `nf-core/diaproteomics` 1.2.2 `main.nf` lines817–821 applies parameterized
peakgroup/global q-value filters during score export. Thus an absent exported ID
is not proof that no internal candidate existed. DIAlignR's exact study overrides
and the cause of each missing/zero/reintegrated value remain unqualified. Preserve
cross-dilution library selection, the twofold versus 1:10 source conflict and
nominal solution concentration versus certified/injected amount distinction.
The JY HLA eluate is the matrix; post-isolation standards do not measure native
processing, MHC capture, precursor abundance or intracellular concentrations.

## Independent verification and execution

Python csv/Decimal generated the inventories and comparisons. A separate pandas
C-parser / Fraction implementation, importing none of those audit modules,
replayed all 29,039,248 literal cells, 12,851 library memberships, 76,561 scored
rows, 64,070 aligned links, 297,876 numerical comparisons and 90 standard slots.
It reproduced both the failed namespace hypothesis and the amended mapping.
[Verification](results/verification.json) passed on its first attempt; every
stage completed without a resource violation. No source or executed code was
rewritten after running.

Explicit E9/F checks of all new frozen audit code pass. Repository lint and
format checks pass over tracked Python files (the repository excludes frozen
experiments from formatting). A whole-working-directory invocation also sees an
untracked historical source under `data/erap1_source_qualification/...`; it fails
only on that archive's formatting. The archive is preserved untouched and is not
part of this publication. [Exact validation scope](results/validation-checks.json).

| Stage | Elapsed seconds | OS child peak MiB |
|---|---:|---:|
| Retrieve 11 assets | 251.608 | 35.0 |
| Structural inventory | 13.520 | 229.2 |
| Feature lineage and original namespace hypothesis | 4.272 | 268.2 |
| Amended exact-form membership | 1.685 | 59.4 |
| Independent complete replay | 49.529 | 783.2 |

Local CPU only; no Modal/GPU, installation or shared-environment change. Each
stage has a 20-minute limit, 3 GiB RSS cap, 1 GiB disk reserve plus 32 MiB startup
margin, 50 ms process-tree polling and OS child peak verification. BLAS/OpenMP
threads are one; `PYTHONDONTWRITEBYTECODE=1`, `MPLBACKEND=Agg`. Each stage records
its actual runtime, environment, child process and log. Original retrieval session
14150 was confirmed absent with successful terminal receipts; no restart occurred.

The exact stage commands, in order, use the frozen launcher:

```sh
sh experiments/2026-10-25_0844_codex_dilution-scoring-lineage/reproduce/launch.sh retrieve
sh experiments/2026-10-25_0844_codex_dilution-scoring-lineage/reproduce/launch.sh inventory
sh experiments/2026-10-25_0844_codex_dilution-scoring-lineage/reproduce/launch.sh lineage
sh experiments/2026-10-25_0844_codex_dilution-scoring-lineage/reproduce/launch.sh membership
sh experiments/2026-10-25_0844_codex_dilution-scoring-lineage/reproduce/launch.sh verify
```

Completed receipts and source destinations refuse overwrite. A fresh replay
requires a new registered directory and explicitly repinned paths, not deleting
old receipts or mutating these source snapshots. `launch.json` retains the
original retrieval registration; the `*-launch.json` files document subsequent
prospective stages/amendments.

The five generated JSON artifacts larger than 1 MiB are published as deterministic
gzip files to keep the PR reviewable. Their original local bytes remain unchanged;
the [compression manifest](reproduce/compressed-artifacts.json) pins both forms.
Fresh execution regenerates the original JSON paths. For verification-only use
from a checkout, restore those paths with the following hash-checked helper; it
refuses to overwrite different local content:

```sh
/Users/iskander/code/shared-virtual-env/bin/python -B experiments/2026-10-25_0844_codex_dilution-scoring-lineage/reproduce/materialize_artifacts.py
```

## Training scope, decision and next action

No model, pretraining, synthetic biological training, optimizer, loss, assay head
or parameter update is part of this audit. The assay-to-output mapping is original
aligned/scored MS quantities to provenance/disposition tables, not an availability
prediction. Validation/test splits, per-example prediction dumps, IC50 regression
and 500 nM metrics are intentionally inapplicable. Original native outer
partitions remain closed. There is no winning predictive condition or RL claim.

Proceed with the shared Hitlist source adapter using explicit ID namespaces,
exact form membership, all original missingness states and source provenance.
The implementation and tested publication remain pending. Calibration training
is still unadmitted: its likelihood/selection policy and remaining protocol
parameters need a separate registered contract. Even a successful bounded
nine-standard dose diagnostic cannot establish broad sequence transfer or native
MS1 calibration. Do not restart the failed native response-only search. The full
precursor-relative APC availability and matched differentiable/discrete modeling
goal remains active and unachieved.
