# Presto

One shared observation model for MHC class-I/class-II affinity and presentation,
with scoped APC state and source-protein context.

## Install and use

From this checkout:

Requires Python 3.11 or newer.

```bash
python -m pip install -e '.[hitlist,dev]'
python -m presto --help
python -c 'import presto; print(presto.__file__, presto.__version__)'
```

The PyPI name `presto` belongs to another project. Use this repository or its
GitHub release wheels. Historical release interfaces and checkpoint readers remain
available; the current workflow uses the shared observation model below.

```bash
presto data verify-corpus --corpus-dir /path/to/corrected-v4
presto data export-presentation --corpus-dir /path/to/corrected-v4 --output-dir exported
presto data prepare-presentation --manifest preparation.json --output-dir prepared
presto train --manifest training.json --preflight
presto train --manifest training.json --run-dir experiments/YOUR_RUN --device cpu
presto evaluate observations --checkpoint selected.pt --manifest training.json --output-dir evaluation
presto predict observations --checkpoint selected.pt --inputs queries.jsonl --output predictions.json
```

Preparation emits pinned source files and a source manifest. A training manifest
adds the explicit model, optimizer, checkpoint-selection and parent-checkpoint
settings; see the [current workflow](docs/current_workflow.md). Evaluation sources
declare their split, allowing validation evaluation to avoid opening TEST files.
Prediction accepts label-free observations and returns endpoint units and support
diagnostics. Neither command imports a frozen experiment runtime.

## Architecture and evidence

- [Current architecture, interfaces and checkpoint decision](docs/current_workflow.md)
- [Model and observation schema](docs/modular_model.md)
- [Training controls and compatibility](docs/unified_training.md)
- [Affinity-conditioned processing](docs/affinity_conditioned_processing.md)
- [Corpus generations and preparation](docs/presentation_data.md)
- [CLI](docs/cli.md)

Class I and II share the encoder, pMHC interaction, APC-state representation and
trainer. Class II enumerates binding registers. Source flanks and cellular context
enter processing/presentation; intrinsic affinity does not consume those inputs.
All supported parameters can train jointly. Independent assay supervision and
held-out checks are necessary to interpret component scores biologically.

The retained broad-presentation checkpoint is the current development default.
The newer boundary-aware processing candidate is provisional. TEST remains sealed;
isolated gene/drug effects and absolute surface-abundance calibration are unproven.
See the workflow page for exact checkpoint hashes and evidence.

## Development

```bash
./lint.sh
./test.sh
python -m build
python scripts/check_wheel.py dist/*.whl
mkdocs build --strict
```

Active runtime code belongs in `models/`, `data/`, `training/`, `inference/`,
`evaluation/` and `cli/`. `legacy/` contains historical implementations.
Shared orchestration belongs in `scripts/`; experiment-specific launchers and
immutable results belong in `experiments/`. New work follows
[the experiment workflow](experiments/EXPERIMENT_WORKFLOW.md).

Keep raw datasets, checkpoints, local build outputs and caches out of source
commits. Preserve their hashes and recovery locations in the experiment registry.
Historical evidence is retained; it does not define the current installed model.
