# Native intracellular source: canonical replay

The new Hitlist native-peptidomics package passes independent original-source
reconciliation. This makes source data queryable for the planned biological fit;
it does not demonstrate predictive quality or calibrated concentration.

- Agent/model: Codex / GPT-6. Registered 2026-10-25, family 0519.
- Presto base: `5e3a8c033df364d8b67aab352c4e8b2f92dc8ef7`, dirty (unrelated ongoing
  todo edits preserved). Hitlist base: `8008fd713bfdb8f340bd635cc830803662c9a186`,
  dirty with a complete changed-source overlay in `reproduce/hitlist-overlay/`.
- Data: 15 exact original PXD018722 Hi20S assets pinned in `reproduce/launch.json`.
  Native intracellular HEK293T control/PSMD2-knockdown signal; not MHC elution,
  professional APC measurements, complete KO or calibrated molarity.
- Training/pretraining: none. No synthetic biological data, objective, loss weights,
  assay-to-model-output mapping, optimizer, train/validation/test split or predictions.
  No held-out outcome inspection: this is source fidelity, not predictive evaluation.
- Hardware: local CPU, no Modal/GPU. Python and package versions are in
  `reproduce/environment.json`. Overrides: `PYTHONDONTWRITEBYTECODE=1` and
  `PYTHONPATH` pointing to the isolated Hitlist checkout; shared installs unchanged.

| Verified evidence | Count |
|---|---:|
| Original peptide / form / feature identities | 10,647 / 10,935 / 40,679 |
| Original quantities, all grains | 255,635 |
| Raw sequence / LFQ sequence values | 74,529 / 63,882 |
| Form / feature intensities | 76,545 / 40,679 |
| Complete deposited reference records | 73,113 |
| All candidate spans (literal / I-L-only) | 38,238 / 403 |
| Explicit original workbook cells | 292,449 |
| Source parent claims and alternatives | 228,749 |
| Source-wide dependency components | 1,689 |
| Retained date-corrupted gene annotations | 13 |

All 16 relation primary keys and all 28 declared relationships pass closure checks.
Every original TSV row, XML node and JSON document is compared to the public source
records. Every source quantity, entity/native-ID mapping, full reference header and
sequence, original workbook cell and parent claim is independently reconciled.
All origins match qualified family0424; graph traversal independently reproduces
all dependencies. Additional form/feature/publisher parent claims merge **zero**
earlier components, so the earlier grouping remains applicable. These counts do
not select a supervised cohort or establish biological independence.

Study metadata now explicitly distinguishes the article's triplicate datasets and
three independent experiments (figure scopes) from unresolved assignment of each
run to a physical specimen or independent experiment. The reported 20S complex
fraction is not a numeric remaining PSMD2 activity. Original raw/LFQ quantities,
sequence/form/feature grain, matching-only evidence, source zeros and missingness
remain separate. Reporter possibilities stay provenance/exclusion evidence.

| Condition / stage | Outcome | Seconds | Peak observed process-tree RSS |
|---|---|---:|---:|
| Public writer plus deterministic original-source replay | Pass | 40.49 | 809.4 MiB |
| Independent public-query reconciliation, corrected verifier | Pass | 29.31 | 1926.3 MiB |
| Initial independent verifier | Failed in verifier FASTA delimiter parser | 25.30 | 1618.1 MiB |

There is no fitted winner; the canonical source projection agrees with the original
source audit. The first monitor could not inspect process trees in the macOS
sandbox; an authoritative process check found no child/package, and a revised
monitor was run with process-inspection access. A recovery invocation with the
wrong working directory failed before mutation and the existing-log guard prevented
a duplicate launch. The first independent verifier split on arrows inside 19
FASTA headers. Production was correct; only its verifier changed to line-start
header delimiters. Failed sources/logs/receipts are retained, and neither data nor
acceptance tolerances changed. See the two recovery notes in `reproduce/`.

The 20-minute/stage, 3 GiB process-tree RSS and 1 GiB disk reserve plus 32 MiB
startup margin remained unchanged. RSS was observed at one-second intervals.
No source downloading or numerical training occurred.

## Reproduction and artifacts

Run `./reproduce/launch.sh build` then `./reproduce/launch.sh verify` from a fresh
registered destination using the pinned interpreter. The launcher refuses existing
stage results and requires the exact Hitlist overlay hashes. Reconstruct the
Hitlist code from its pinned base plus `reproduce/hitlist-overlay/`; do not rely on
a later shared editable checkout. Source snapshots and exact executed launcher
files are retained. macOS execution needs process-inspection access for the guard.

The canonical package remains once at:
`/Users/iskander/code/presto/.worktrees/presto-modular-model/data/native_peptidomics_packages/2026-10-25_0519_codex_native-source-canonical`.
`results/package-manifest.json` records every package file hash and size. Original
source paths remain in the launch contract; staging symlinks avoid recopying raw
inputs, and the package writer uses independent copy-on-write snapshots where
supported. Large original assets and the package are intentionally not committed.
`results/independent-verification.json` and runtime receipts contain the complete
closure counts and measured resources. Exact non-default commands are in the
runtime receipts; launch/source manifests do not claim an unmodified checkout.

Hitlist [PR839](https://github.com/pirl-unc/hitlist/pull/839), commit
`bfbaa67fa87300f5ec4fbadad0ea728257cd1738`, contains the adapter and verification.
Format, lint and all 3,533 default non-integration tests pass; hosted tests and
tested PyPI release remain pending.

Next: finish those release gates, then a versioned explicit
eligibility/contrast recipe and the planned biological fit. The full availability
and exact-versus-RL objective remains open. This result resolves source ingestion,
not biological transfer, absolute concentration, or the earlier IC50 underfit.
