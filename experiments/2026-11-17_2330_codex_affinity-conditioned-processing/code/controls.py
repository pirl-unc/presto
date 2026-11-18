"""Fixed-checkpoint APC/flank input sensitivity; validation only, no refitting."""
import gzip
import json
import time
from pathlib import Path

import modal
from common import NAME, BROAD, PRIOR, E, VOLUME, rows, sha, dump
from launch import image, GPU

app = modal.App('presto-processing-input-controls')
MODES = ('original', 'apc-unknown', 'apc-shuffle', 'flank-unknown', 'flank-shuffle')


@app.function(image=image, gpu=GPU, cpu=4, memory=65536, timeout=14400,
              volumes={'/volume': VOLUME}, single_use_containers=True, retries=0)
def evaluate_controls(run):
    import torch
    from presto.data.modular import Observation
    from presto.inference.modular import ModularPredictor
    from presto.training.modular_evaluation import evaluate
    from project_context import Projection
    from evaluate import supply_readouts
    from apc_controls import controlled as apc_control
    from flank_controls import controlled as flank_control
    from fit import PARENT, PARENT_SHA

    root = Path('/volume') / NAME
    terminal = run.endswith('-terminal')
    fit_run = run.removesuffix('-terminal')
    path = PARENT if run == 'parent' else root / 'fits' / fit_run / 'raw' / ('last.pt' if terminal else 'selected.pt')
    if run == 'parent':
        assert sha(path) == PARENT_SHA
    evaluation_receipt = json.loads((root / 'evaluation' / run / 'receipt.json').read_text())
    assert evaluation_receipt['checkpoint_sha256'] == sha(path)
    out = root / 'controls' / run
    if (out / 'receipt.json').exists():
        report = json.loads((out / 'receipt.json').read_text())
        assert report['checkpoint_sha256'] == sha(path)
        return {'run': run, 'cached': True}
    if out.exists():
        out.rename(out.with_name(run + '-incomplete-' + str(time.time_ns())))
    out.mkdir(parents=True)
    start = time.monotonic()
    predictor = ModularPredictor(path, device='cuda')
    model = predictor.model.eval()
    report = {
        'run': run, 'checkpoint': str(path), 'checkpoint_sha256': sha(path),
        'test_accessed': False, 'files': [], 'populations': {},
        'hardware': {'requested_gpu': GPU, 'name': torch.cuda.get_device_name(),
                     'memory_bytes': torch.cuda.get_device_properties(0).total_memory},
        'scope': 'fixed-checkpoint inference association/sensitivity; not trained-shuffle or isolated causal intervention',
    }
    populations = {k: Path('/volume') / BROAD / 'inputs' / f'{k}-validation.jsonl' for k in ('apc', 'expanded')}
    populations['matched'] = Path('/volume') / PRIOR / 'inputs/matched-validation-v3.jsonl'
    populations['rank-matched'] = root / 'inputs/rank-matched-validation.jsonl'
    for population, source in populations.items():
        originals = list(rows(source))
        report['populations'][population] = {'source_sha256': sha(source), 'modes': {}}
        for mode in MODES:
            function = flank_control if mode.startswith('flank-') else apc_control
            transform = 'original' if mode == 'original' else 'unknown' if mode.endswith('unknown') else 'within-study-shuffle'
            transformed, receipt = function(originals, transform)
            projection = Projection(model.config.to_dict())
            observations = [Observation.from_dict(projection.record(r)[0]) for r in transformed]
            assert all(r.split == 'validation' for r in observations)
            metrics, records = evaluate(model, observations, list(range(len(observations))), predictor.scales, batch_size=64)
            for record in records:
                supply_readouts(record)
            prefix = f'{population}-{mode}'
            filename = prefix + '-predictions.jsonl.gz'
            with gzip.open(out / filename, 'wt') as stream:
                for record in records:
                    stream.write(json.dumps(record, separators=(',', ':')) + '\n')
            dump(out / (prefix + '-metrics.json'), metrics)
            report['files'].extend([filename, prefix + '-metrics.json'])
            receipt['projection'] = projection.receipt()
            report['populations'][population]['modes'][mode] = receipt
            dump(out / 'progress.json', report)
            VOLUME.commit()
    report['seconds'] = time.monotonic() - start
    dump(out / 'receipt.json', report)
    VOLUME.commit()
    return {'run': run, 'seconds': report['seconds']}


@app.function(image=image, cpu=2, memory=4096, timeout=43200, volumes={'/volume': VOLUME})
def coordinate(runs):
    root = Path('/volume') / NAME / 'calls'
    key = '__'.join(runs)
    journal = root / f'controls-{key}.json'
    if journal.exists():
        entries = json.loads(journal.read_text())
        calls = [(r['run'], modal.FunctionCall.from_id(r['call'])) for r in entries]
    else:
        calls = [(run, evaluate_controls.spawn(run)) for run in runs]
        dump(journal, [{'run': run, 'call': c.object_id} for run, c in calls])
        VOLUME.commit()
    results, errors = [], []
    for run, call in calls:
        try:
            results.append(call.get())
        except Exception as error:
            errors.append({'run': run, 'error': repr(error)})
        dump(root / f'controls-{key}-progress.json', {'results': results, 'errors': errors})
        VOLUME.commit()
    if errors:
        raise ValueError(errors)
    return results


@app.local_entrypoint()
def main(runs: str):
    call = coordinate.spawn(runs.split(','))
    dump(E / 'reproduce' / f'controls-{runs}-call.json', {'call': call.object_id, 'runs': runs})
    result = call.get()
    dump(E / 'results' / f'controls-{runs}-completion.json', result)
    print(json.dumps(result), flush=True)
