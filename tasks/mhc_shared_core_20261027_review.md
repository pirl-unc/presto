# Common-core integration review

Codex / GPT-6, 2026-10-27. Local branch
`codex/shared-core-assumptions-20261027`, based on immutable `c0358852`.

## Outcome

The MHC-II work develops Presto's shared **modular research path**. MHC-I,
MHC-II, bulk MS and component observations use the common observation schema,
sequence encoder, relevant feature blocks, trainer and checkpoint APIs. Whole
class-I peptides and class-II registers remain explicit candidate constructions.
The legacy default model is a separate existing path; this audit does not claim
that it has been migrated or that these changes have been merged.

The reusable gap was molecular-reference provenance. The DO experiment manually
selected an assumed alpha because the common catalog adapter labeled every
selection as reported. `HumanMHCCatalog.resolve` now accepts a typed
`MHCReferenceAssumption` for either class. The caller supplies the modeled
restriction, reason and evidence reference, while the original argument retains
actual reported typing, including partial typing or unknown. Reported alleles
cannot be replaced, catalog checks remain active, and strict callers behave as
before. The model and trainer gain no source-specific branch or new network.

The implementation reuses `MHCInput` and its existing diagnostics/serialization;
there is no additional observation wrapper or prediction API. Source adapters
translate evidence, while studies supply configurations and comparison controls.
Future native reference-assumption experiments should use this shared resolver.
Completed experimental source snapshots retain the code that actually ran.

## Evidence

- **244 tests passed** across all 16 `test_modular*.py` files in 16.02 seconds,
  one CPU/Torch thread. [JUnit report](mhc_shared_core_20261027_tests.xml).
- Both MHC classes make gradient-driven updates to the **same** encoder,
  pMHC/context/MS feature blocks and quantitative readouts, using one optimizer.
  Class-II register pooling receives gradients only where appropriate.
- A single canonical `fit_index` job admits both classes and bulk quantities,
  passes parameter/feature coverage, saves one checkpoint and exactly replays
  its saved test predictions through `ModularPredictor` with matched batching.
- Cellular ordinal inputs change cellular outputs but leave purified affinity,
  receptor binding, digestion and the shared MS detector invariant in both classes.
- Explicit assumption provenance survives observation serialization and public
  prediction after checkpoint reload. Altering provenance alone leaves the score
  exactly unchanged. Missing metadata, conflicting typing and invalid catalog
  resolution are rejected.
- The [input-only replay](mhc_shared_core_20261027_replay.json) loads the full
  pinned catalog, reproduces the DO experiment's exact two groove sequences and
  preserves unknown source alpha identity. It verifies all **168 frozen files**
  against the completed experiment's manifest. No fit or prediction was rerun.
- Repository-wide Ruff check/format (424 files) and Git whitespace checks pass.
  An inherited helper formatting error was repaired with exact executed bytes
  retained and AST equality verified; see the
  [repair receipt](mhc_shared_core_20261027_archive/formatting-repair.json).
  [Verification hashes](mhc_shared_core_20261027_verification.json) identify the
  tested source and retained evidence. No dependencies were installed or changed.

The first focused run identified two fixture mistakes, documented in the
[spec](mhc_shared_core_20261027_spec.md): mixed inference kernels and unsupported
test-only flanking residues. Fixtures were corrected without relaxing production
guards or changing scientific data; the final suite passes without skips.

## Predictive limits

Shared software is established; **MHC-I predictive improvement is not measured**.
A future transfer claim requires matched held-out MHC-I evaluation and a control
without the added MHC-II training, including checks for negative transfer. The
native DO:DM development comparison remains negative: weighted test RMSE was
0.5765 versus 0.5281 for condition means and 0.5070 for the shuffled control.
This integration does not qualify that endpoint, identify processing components,
promote a baseline or satisfy the remaining scientific milestones for PR #131.

## Reproduce the software checks

From this checkout's `tests/` directory, set its parent as `PYTHONPATH`, then:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/private/tmp/presto-shared-core-20261027 \
/Users/iskander/code/shared-virtual-env/bin/python - <<'PY'
from pathlib import Path
import pytest
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
files = sorted(str(path) for path in Path('.').glob('test_modular*.py'))
raise SystemExit(pytest.main([
    *files, '--noconftest', '--confcutdir=.', '--import-mode=importlib',
    '-p', 'no:cacheprovider', '-o', 'addopts=', '-q',
]))
PY
```

From the checkout root, run `tasks/verify_mhc_shared_core_20261027.py` with the
completed experiment directory and a new output JSON path as its two arguments.
The recorded source directory is in the replay receipt. It requires the pinned
catalog at the original recorded path and reads all frozen inputs without
modifying them. These are software integration checks, not a new biological
experiment or a substitute for the canonical experiment registry.
