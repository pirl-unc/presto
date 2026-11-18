# Run the shared presentation model

The inference helper uses the canonical implementation frozen in
`reproduce/runtime-pathway/` (core5f9bfe37), through `reproduce/inference-runtime/presto`. It does not
depend on whichever Presto checkout is installed in the shared environment.
Both MHC classes use the same predictor and checkpoint.

**Working default:** `expanded-state-sqrt-s42-n6000`, selected at update 6,000.
The recorded SHA256 is `1b6011396a563f52ea1d6da4bb1d4e69b8dbf0103148ba829225581c4d312f68`.
All 12 fits and selected/terminal evaluations are complete. This improves ordinary
presentation in both classes on development validation; affinity-matched class I
regresses and TEST remains sealed. See [DECISION.md](DECISION.md).

From this experiment directory, using the existing shared Python environment:

```sh
python code/predict_example.py
python code/predict_example.py --inputs my-queries.jsonl --output my-predictions.json
```

The default verifies `results/preferred-checkpoint.json` and the checkpoint hash.
The existing local experiment already contains the checkpoint. In a fresh source
checkout, retrieve it and its exact fit receipts from the authorized Modal volume:

```sh
python code/collect.py --runs expanded-state-sqrt-s42-n6000
python code/predict_example.py
```

For a different explicit checkpoint, add `--checkpoint /absolute/path/to/selected.pt`.
If local DNS is unavailable, the optional process-local verified transport wrapper
can invoke the same collector: `python code/dns_transport.py code/collect.py --runs expanded-state-sqrt-s42-n6000`.
It does not change system settings, credentials or packages.

`results/example-queries.jsonl` supplies four label-free requests: presentation
and IC50 for class I and class II. Queries use canonical `Observation` fields,
`split: predict`, `value: null`, typed MHC groove sequences and explicit candidate
sets. Presentation returns logits and ranking scores; affinity also returns
decoded nM. Predictions include input-support and component diagnostics.

The public API is:

```python
from presto.data.modular import Observation
from presto.inference.modular import ModularPredictor

predictor = ModularPredictor("selected.pt", device="cpu")
results = predictor.predict([Observation.from_dict(query)])
```

Unrepresented APC categories fail by default. If the intended behavior is to
predict with those categories unknown, use the explicit policy:

```python
results = predictor.predict(rows, unsupported_apc_policy="unknown")
```

The CLI exposes the same behavior with `--unsupported-apc-policy unknown`.
Omitted categorical claims appear in `unrepresented_apc_context`; the predictor
does not invent a control or substitute a different cell type. Numeric units
and unsupported measured-expression features still require their declared
contracts. Current fits contain cell identity, cytokine and intervention
annotations, but no matched quantitative expression training targets.

Cell state feeds a shared 64-dimensional APC representation and class-scoped
supply/selection terms. Intrinsic affinity has no APC input. The sequence,
affinity, processing and APC parameters are all trained jointly. Class II
evaluates candidate nine-residue registers; class I uses the full peptide.

NetMHCpan and NetMHCIIpan are not inference dependencies. In this experiment,
their frozen BA predictions define the separate affinity-matched evaluation.
Native MS detections supply positive labels; unobserved same-source windows
supply sampled comparison labels. The output is a presentation ranker under
that objective, not absolute surface abundance or calibrated knockout severity.

The sealed TEST split has not been used. Exact training configuration, source
hashes, validation projections, checkpoint and coverage receipts live under
`results/fits/<run>/`; the experiment README records measured performance and
limitations.

The current shared architecture, input meanings and ablation definitions are in [ARCHITECTURE.md](ARCHITECTURE.md).

The completed no-state square-root baseline is also verified through this CLI on CPU:
`results/fits/expanded-none-sqrt-s42-n6000/raw/selected.pt`, SHA256
`df53b566ea0cfbf310bea7c2872ffa97aec064172bec4fdd2b6fe0a6f99b86a1`.
Use `--unsupported-apc-policy unknown` for the example requests because this control
has no APC module. Its four label-free responses are preserved in
`results/inference-sqrt-none42-example.json`. This is a diagnostic no-state baseline; the default above uses shared APC state.
