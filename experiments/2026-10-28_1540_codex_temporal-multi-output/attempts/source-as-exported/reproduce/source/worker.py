"""One canonical shared model/trainer, live validation and complete held-out outputs."""

from collections import Counter, defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import platform
import subprocess
import sys
import time

FID = '2026-10-28_1540_codex_temporal-multi-output'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    p = path.with_suffix(path.suffix + '.tmp')
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    p.replace(path)


def compress(source, dest):
    source, dest = Path(source), Path(dest)
    if dest.exists():
        try:
            digest = hashlib.sha256()
            with gzip.open(dest, 'rb') as stream:
                for block in iter(lambda: stream.read(1 << 20), b''):
                    digest.update(block)
        except (EOFError, OSError):
            dest.rename(dest.with_name(dest.name + '.interrupted-' + str(time.time_ns())))
        else:
            assert digest.hexdigest() == sha(source), 'completed prediction archive changed'
            return
    temporary = dest.with_suffix(dest.suffix + '.tmp')
    with source.open('rb') as a, temporary.open('wb') as b:
        with gzip.GzipFile(fileobj=b, mode='wb', mtime=0) as z:
            for block in iter(lambda: a.read(1 << 20), b''):
                z.write(block)
    temporary.replace(dest)


def ligand_reports(records):
    from presto.training.modular_evaluation import summarize, summarize_strata, binary_metrics

    el = [r for r in records if r['endpoint'] == 'elution_observation']
    reported = [r for r in el if r['source'] != 'hitlist_ligand_background']
    populations = {}
    for kind in ('same_protein', 'reference_proteome'):
        negatives = [r for r in el if r['observation']['provenance'].get('background_kind') == kind]
        parents = {r['observation']['provenance']['parent_observation_id'] for r in negatives}
        positives = [r for r in el if r['observation_id'] in parents]
        cohort = positives + negatives
        if cohort:
            populations[kind] = dict(
                metrics=summarize(cohort),
                by_study=summarize_strata(cohort, fields=('mhc_class', 'study')),
                length_only=binary_metrics(
                    [r['target'] for r in cohort],
                    [len(r['peptide']) for r in cohort],
                    [len(r['peptide']) >= 10 for r in cohort],
                ),
                n_parents=len(parents),
            )
    return dict(
        source_reported_by_study=summarize_strata(reported, fields=('mhc_class', 'study'))
        if reported
        else [],
        backgrounds=populations,
        interpretation=(
            'reported outcomes are assay-scoped; generated backgrounds are unobserved, '
            'with replacement; additive components are not independently identified'
        ),
    )


def run(condition, output, *, resume=False):
    sys.path.insert(0, '/runtime')
    import psutil
    import torch
    import modal
    from checkpoint_observer import save as save_observed, restore as restore_observed
    from presto.data.modular import Observation
    from presto.data.modular_sources import ObservationIndex
    from presto.models.modular import MODEL_VERSION
    from presto.training import modular_run as runner
    from presto.training.modular import load_checkpoint
    from presto.training.modular_evaluation import evaluate, summarize, summarize_strata

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'status.json').exists() and not resume:
        raise ValueError('Already attempted; explicit verified resume required')
    pins = read('/bundle/runtime-pins.json')
    for relative, digest in pins['files'].items():
        assert sha(Path('/runtime/presto') / relative) == digest, relative
    root = Path('/runs') / FID
    prepared = read(root / 'prepared/prepared.json')
    spec = prepared['manifests'][condition]
    assert sha(spec['path']) == spec['sha256']
    binding = dict(
        condition=condition,
        manifest_sha256=spec['sha256'],
        runtime_pins_sha256=sha('/bundle/runtime-pins.json'),
        worker_sha256=sha(__file__),
        observer_sha256=sha(Path(__file__).with_name('checkpoint_observer.py')),
    )
    restored = restore_observed(output / 'raw/resume.pt', binding) if resume else None
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    assert torch.cuda.is_available()
    gpu = torch.cuda.get_device_properties(0)
    assert 'H100' in gpu.name
    torch.cuda.reset_peak_memory_stats()
    hardware = dict(
        requested_gpu='H100!',
        actual_gpu=gpu.name,
        total_gpu_bytes=gpu.total_memory,
        python=sys.version,
        torch=str(torch.__version__),
        numpy=__import__('numpy').__version__,
        cuda=torch.version.cuda,
        platform=platform.platform(),
        threads=4,
        interop_threads=1,
        tf32=False,
        amp=False,
        compile=False,
        model_version=MODEL_VERSION,
        source=pins,
        nvidia_smi=subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            text=True,
        ).strip(),
    )
    write(output / 'hardware.json', hardware)
    write(output / 'status.json', dict(status='admitting_frozen_manifest', condition=condition))
    start = time.monotonic()
    index = ObservationIndex(spec['path'])
    state = dict(
        steps=0, examples=0, optimizer_seconds=0.0, endpoint_class_examples=Counter(), peak_rss=0
    )
    original = runner.train_step
    original_evaluate = runner.evaluate
    process = psutil.Process()
    seen = set()
    seen_at_validation = {}
    replay_rows = defaultdict(dict)
    elapsed_before = 0.0
    if restored:
        for key in state:
            state[key] = restored[key]
        state['endpoint_class_examples'] = Counter(state['endpoint_class_examples'])
        seen.update(restored['seen'])
        seen_at_validation.update(restored['seen_at_validation'])
        for item in restored['replay_rows']:
            replay_rows[tuple(item['key'])] = {
                value['observation_id']: Observation.from_dict(value) for value in item['rows']
            }
        elapsed_before = restored['elapsed_seconds']
    original_save = runner.save_resume

    def observer_state():
        return dict(
            **{k: v for k, v in state.items() if k != 'endpoint_class_examples'},
            endpoint_class_examples=dict(state['endpoint_class_examples']),
            seen=list(seen),
            seen_at_validation=dict(seen_at_validation),
            replay_rows=[
                dict(key=list(key), rows=[r.to_dict() for r in rows.values()])
                for key, rows in replay_rows.items()
            ],
            elapsed_seconds=elapsed_before + time.monotonic() - start,
        )

    def observed_snapshot(path, *args, **kwargs):
        save_observed(
            original_save,
            path,
            args,
            kwargs,
            observer_state,
            binding,
            modal.Volume.from_name('presto-checkpoints').commit,
        )

    runner.save_resume = observed_snapshot
    write(
        output / ('attempt-' + str(time.time_ns()) + '.json'),
        dict(
            resumed=resume,
            restored_steps=state['steps'],
            started_unix=time.time(),
            binding=binding,
        ),
    )

    def observed(model, optimizer, rows, scales, **kwargs):
        torch.cuda.synchronize()
        before = time.monotonic()
        result = original(model, optimizer, rows, scales, **kwargs)
        torch.cuda.synchronize()
        state['optimizer_seconds'] += time.monotonic() - before
        state['steps'] += 1
        state['examples'] += len(rows)
        seen.update(r.observation_id for r in rows)
        for row in rows:
            state['endpoint_class_examples'][row.endpoint + ':' + row.mhc[0].mhc_class] += 1
            key = (row.endpoint, row.mhc[0].mhc_class)
            if len(replay_rows[key]) < 4:
                replay_rows[key][row.observation_id] = row
        if state['steps'] % 1000 == 0 or state['steps'] == spec['max_steps']:
            seen_at_validation[state['steps']] = len(seen)
        state['peak_rss'] = max(state['peak_rss'], process.memory_info().rss)
        if state['steps'] == 1 or state['steps'] % 25 == 0:
            point = dict(
                **state,
                max_steps=spec['max_steps'],
                unique_training_rows_seen=len(seen),
                latest_training_loss=result['loss'],
                elapsed_seconds=elapsed_before + time.monotonic() - start,
                updated_unix=time.time(),
                gpu_allocated=torch.cuda.max_memory_allocated(),
                gpu_reserved=torch.cuda.max_memory_reserved(),
            )
            write(output / 'progress.json', point)
            with (output / 'progress.jsonl').open('a') as stream:
                stream.write(json.dumps(point) + '\n')
        return result

    runner.train_step = observed

    def observed_evaluate(*args, **kwargs):
        metrics, records = original_evaluate(*args, **kwargs)
        if records and records[0]['split'] == 'validation':
            write(
                output / 'live-evaluation.json',
                dict(
                    steps=state['steps'],
                    metrics=metrics,
                    ligands=ligand_reports(records),
                    updated_unix=time.time(),
                ),
            )
        return metrics, records

    runner.evaluate = observed_evaluate
    write(
        output / 'status.json',
        dict(status='training', condition=condition, started_unix=time.time()),
    )
    try:
        summary = runner.fit_index(index, output / 'raw', device='cuda', resume=resume)
        assert summary['steps'] == spec['max_steps']
        assert len(seen) == len(index.partitions['train:assay'])
        write(
            output / 'status.json',
            dict(status='summarizing_heldout_predictions', condition=condition),
        )
        model, scales, payload = load_checkpoint(output / 'raw/selected.pt', device='cuda')
        model.eval()
        checkpoint_hash = sha(output / 'raw/selected.pt')
        report = dict(
            condition=condition,
            summary=summary,
            checkpoint=str(output / 'raw/selected.pt'),
            checkpoint_sha256=checkpoint_hash,
            selected_steps=payload['metadata']['steps'],
            selected_unique_training_rows_seen=seen_at_validation[payload['metadata']['steps']],
            parameters=sum(p.numel() for p in model.parameters()),
            hardware=hardware,
            metrics_by_era={},
            metrics_by_assay_study={},
            ligands={},
            overall_losses={},
            training_exposures=dict(state['endpoint_class_examples']),
            unique_training_rows_seen=len(seen),
            all_training_rows_exposed=True,
        )
        for split in ('train', 'validation', 'test'):
            source = output / 'raw' / f'{split}-predictions.jsonl'
            compress(source, output / f'{split}-predictions.jsonl.gz')
            if split == 'train':
                continue
            with source.open() as stream:
                rows = [json.loads(line) for line in stream]
            report['overall_losses'][split] = sum(r['loss'] for r in rows) / len(rows)
            report['metrics_by_era'][split] = {}
            for era in ('2026', 'older_or_mixed'):
                subset = [
                    r for r in rows if r['observation']['provenance']['evaluation_era'] == era
                ]
                report['metrics_by_era'][split][era] = summarize(subset)
            report['metrics_by_assay_study'][split] = summarize_strata(rows)
            report['ligands'][split] = ligand_reports(rows)
            host_groups = defaultdict(list)
            for row in rows:
                raw = row['observation']['provenance'].get('raw_record', {})
                host_groups[
                    (
                        str(raw.get('host') or 'unknown'),
                        str(raw.get('source_organism') or 'unknown'),
                    )
                ].append(row)
            write(
                output / f'{split}-host-source-strata.json',
                dict(
                    source_interpretation='Reported source organism, or positive parent source '
                    'organism for generated backgrounds; background proteome is separately pinned',
                    strata=[
                        dict(host=h, source_organism=s, metrics=summarize(v))
                        for (h, s), v in sorted(host_groups.items())
                    ],
                ),
            )
            del rows
        # Serving replay uses TRAIN examples, never a second pass over held-out data.
        chosen = [row for key in sorted(replay_rows) for row in replay_rows[key].values()]
        _, replay = evaluate(model, chosen, list(range(len(chosen))), scales, batch_size=64)
        with gzip.open(output / 'training-replay.jsonl.gz', 'wt') as stream:
            for row in replay:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        # Exercise public serving with the checkpoint's support guards enabled.
        # This reuses TRAIN probes only; no second held-out inference is performed.
        from dataclasses import replace
        import numpy as np
        from presto.inference.modular import ModularPredictor

        predictor = ModularPredictor(output / 'raw/selected.pt', device='cuda')
        served = predictor.predict(
            [
                replace(row, value=None, unit=None, split='predict', identity_labels={})
                for row in chosen
            ]
        )
        expected = np.array([row['prediction'] for row in replay])
        actual = np.array(
            [row['logit'] if 'logit' in row else row['transformed_value'] for row in served]
        )
        assert [row['observation_id'] for row in served] == [
            row['observation_id'] for row in replay
        ]
        assert all('support' in row['diagnostics'] for row in served)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        report['serving_replay'] = dict(
            status='passed',
            n=len(chosen),
            split='train',
            support_guards_enabled=True,
            maximum_absolute_error=float(np.max(np.abs(actual - expected))),
            rtol=2e-5,
            atol=2e-5,
        )
        assert sha(output / 'raw/selected.pt') == checkpoint_hash
        report['resources'] = dict(
            wall_seconds=elapsed_before + time.monotonic() - start,
            optimizer_seconds=state['optimizer_seconds'],
            peak_rss=state['peak_rss'],
            peak_gpu_allocated=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved=torch.cuda.max_memory_reserved(),
        )
        write(output / 'report.json', report)
        write(
            output / 'status.json',
            dict(status='completed', condition=condition, finished_unix=time.time()),
        )
    except BaseException as exc:
        write(output / 'status.json', dict(status='failed', condition=condition, error=repr(exc)))
        raise
    finally:
        runner.train_step = original
        runner.evaluate = original_evaluate
        runner.save_resume = original_save


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], resume='--resume' in sys.argv[3:])
