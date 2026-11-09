"""Durable preparation, two-GPU queue and common-trainer assay comparison."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import modal

FID = '2026-10-30_1154_codex_direct-factorized-affinity'
PARENT = '2026-10-29_2303_codex_assay-measurement-layer'
GPU = os.environ.get('PRESTO_MODAL_GPU', 'H100!')
assert GPU == 'H100!'
REQUIREMENTS = [
    'torch==2.14.1',
    'numpy==2.5.4',
    'pyyaml==6.0.3',
    'mhcgnomes==3.64.6',
    'mhcseqs==2.6.13',
    'tqdm==4.70.1',
    'psutil==7.2.2',
]
app = modal.App('presto-direct-factorized-20261030')
volume = modal.Volume.from_name('presto-checkpoints')
live = modal.Dict.from_name('presto-cross-class-live')
image = (
    modal.Image.debian_slim(python_version='3.12')
    .uv_pip_install(*REQUIREMENTS)
    .env(
        {
            'OMP_NUM_THREADS': '4',
            'OPENBLAS_NUM_THREADS': '4',
            'MKL_NUM_THREADS': '4',
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONPATH': '/runtime',
        }
    )
)
if modal.is_local():
    F = Path(__file__).resolve().parents[1]
    frozen = json.loads((F / 'reproduce/launch.json').read_text())
    for relative, expected in frozen['experiment_files'].items():
        assert hashlib.sha256((F / relative).read_bytes()).hexdigest() == expected, relative
    image = image.add_local_dir(frozen['runtime_snapshot'], remote_path='/runtime/presto')
    image = image.add_local_dir(F / 'reproduce', remote_path='/bundle')
    image = image.add_local_file(F / 'code/prepare.py', remote_path='/bundle/prepare.py')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_runtime():
    contract = read('/bundle/launch.json')
    for name, expected in contract['runtime_files'].items():
        assert sha(Path('/runtime/presto') / name) == expected, name
    assert sha('/bundle/prepare.py') == contract['experiment_files']['code/prepare.py']
    return contract


def publish(condition, **values):
    value = dict(condition=condition, updated_unix=time.time(), **values)
    live.put(FID + ':' + condition, value)
    return value


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=14400,
    ephemeral_disk=524288,
    volumes={'/volume': volume},
)
def prepare():
    verify_runtime()
    sys.path.insert(0, '/bundle')
    from prepare import prepare as project

    durable = Path('/volume') / FID / 'prepared'
    if (durable / 'prepared.json').exists():
        result = read(durable / 'prepared.json')
        for source in result['sources']:
            assert (
                sha(Path(result['source_volume_root']) / Path(source['path']).name)
                == source['sha256']
            )
        return result
    publish('coordinator', status='preparing_verified_cohort')
    destination = Path('/runs') / FID / 'prepared'
    result = project(
        read('/bundle/parent-manifest.json'),
        read('/bundle/parent-prepared.json'),
        destination,
        Path('/volume') / PARENT / 'prepared',
    )
    shutil.copytree(destination, durable, dirs_exist_ok=True)
    volume.commit()
    publish('coordinator', status='cohort_prepared', preparation=result)
    return result


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=65536,
    timeout=86400,
    ephemeral_disk=524288,
    max_containers=2,
    single_use_containers=True,
    volumes={'/volume': volume},
    retries=0,
)
def fit(condition):
    contract = verify_runtime()
    assert condition in contract['conditions']
    import torch
    from presto.data.modular_sources import ObservationIndex
    from presto.training import modular_run as runner
    from presto.training.modular_evaluation import summarize, summarize_strata

    durable = Path('/volume') / FID
    output = durable / condition
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'receipt.json').exists():
        receipt = read(output / 'receipt.json')
        assert sha(Path('/volume') / receipt['archive']) == receipt['sha256']
        return receipt
    publish(condition, status='staging')
    prepared = read(durable / 'prepared/prepared.json')
    manifest_record = prepared['manifests'][condition]
    manifest_path = Path(manifest_record['path'])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(durable / 'prepared' / manifest_path.name, manifest_path)
    assert sha(manifest_path) == manifest_record['sha256']
    for source in read(manifest_path)['sources']:
        target = Path(source['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(prepared['source_volume_root']) / target.name, target)
        assert sha(target) == source['sha256']
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    gpu = torch.cuda.get_device_properties(0)
    assert 'H100' in gpu.name
    hardware = dict(
        requested_gpu=GPU,
        actual_gpu=gpu.name,
        total_gpu_bytes=gpu.total_memory,
        torch=str(torch.__version__),
        python=sys.version,
        cuda=torch.version.cuda,
        nvidia_smi=subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            text=True,
        ).strip(),
    )
    write(output / 'hardware.json', hardware)
    write(output / 'launch.json', contract)
    index = ObservationIndex(manifest_path)
    raw = output / 'raw'
    resume = (raw / 'resume.pt').exists()
    steps = 0
    if resume:
        checkpoint = torch.load(raw / 'resume.pt', map_location='cpu', weights_only=True)
        steps = checkpoint['metadata']['resume']['state']['steps']
        del checkpoint
    original_step, original_eval = runner.train_step, runner.evaluate
    state = dict(steps=steps, max_steps=4000, hardware=hardware)
    started = time.monotonic()

    def observed_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        state['steps'] += 1
        state['loss'] = result['loss']
        state['elapsed_seconds'] = time.monotonic() - started
        if state['steps'] == 1 or state['steps'] % 25 == 0:
            write(output / 'progress.json', state)
            publish(condition, status='training', **state)
        if state['steps'] % 500 == 0:
            volume.commit()
        return result

    def observed_eval(model, indexed, indices, scales, **kwargs):
        metrics, records = original_eval(model, indexed, indices, scales, **kwargs)
        split = indexed[indices[0]].split if indices else 'empty'
        state[split] = metrics
        write(output / 'progress.json', state)
        publish(condition, status='evaluated_' + split, **state)
        return metrics, records

    runner.train_step, runner.evaluate = observed_step, observed_eval
    try:
        summary = runner.fit_index(index, raw, device='cuda', resume=resume)
    except Exception as exc:
        publish(condition, status='failed', error=repr(exc), **state)
        volume.commit()
        raise
    finally:
        runner.train_step, runner.evaluate = original_step, original_eval
    report = dict(
        condition=condition,
        summary=summary,
        hardware=hardware,
        manifest_sha256=sha(manifest_path),
        checkpoint_sha256=sha(raw / 'selected.pt'),
        original_vs_additional={},
        by_study={},
        ligands={},
    )
    for split in ('validation', 'test'):
        path = raw / (split + '-predictions.jsonl')
        records = [json.loads(line) for line in path.open()]
        report['by_study'][split] = summarize_strata(records)
        ligand = [r for r in records if r['endpoint'] == 'elution_observation']
        report['ligands'][split] = {}
        for kind in ('same_protein', 'reference_proteome'):
            negatives = [
                r for r in ligand if r['observation']['provenance'].get('background_kind') == kind
            ]
            parents = {r['observation']['provenance']['parent_observation_id'] for r in negatives}
            positives = [r for r in ligand if r['observation_id'] in parents]
            if negatives:
                report['ligands'][split][kind] = summarize(positives + negatives)
        report['original_vs_additional'][split] = {
            label: summarize(
                [
                    r
                    for r in records
                    if bool(r['observation']['provenance'].get('assay_holdout_kind')) == additional
                ]
            )
            for label, additional in (('original', False), ('additional_fluorescence', True))
        }
        with (
            path.open('rb') as source,
            gzip.open(output / (split + '-predictions.jsonl.gz'), 'wb') as target,
        ):
            shutil.copyfileobj(source, target)
    write(output / 'report.json', report)
    archive = durable / (condition + '-review.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        for name in (
            'report.json',
            'hardware.json',
            'launch.json',
            'validation-predictions.jsonl.gz',
            'test-predictions.jsonl.gz',
        ):
            tar.add(output / name, arcname=condition + '/' + name)
        for name in (
            'summary.json',
            'history.json',
            'coverage.json',
            'selected.pt',
            'admission.json',
            'resolved-config.json',
        ):
            tar.add(raw / name, arcname=condition + '/raw/' + name)
    receipt = dict(
        condition=condition,
        archive=str(archive.relative_to('/volume')),
        sha256=sha(archive),
        bytes=archive.stat().st_size,
        report=report,
    )
    write(output / 'receipt.json', receipt)
    volume.commit()
    publish(condition, status='finished', **state, result=receipt)
    return receipt


@app.function(image=image, cpu=0.25, memory=1024, timeout=86400, volumes={'/volume': volume})
def coordinate():
    contract = verify_runtime()
    result = prepare.remote()
    pending = dict(contract['wait_for_parent_calls'])
    while pending:
        for name, call_id in list(pending.items()):
            try:
                modal.FunctionCall.from_id(call_id).get(timeout=0)
            except modal.exception.FunctionTimeoutError:
                raise
            except TimeoutError:
                continue
            del pending[name]
        publish(
            'coordinator',
            status='waiting_for_parent_gpus' if pending else 'launching',
            pending_parent_calls=pending,
            preparation=result,
        )
        if pending:
            time.sleep(30)
    receipts = {}
    for mode in ('direct',):
        calls = {f'{mode}-s{seed}': fit.spawn(f'{mode}-s{seed}') for seed in (42, 43)}
        write(
            Path('/volume') / FID / (mode + '-calls.json'),
            {key: call.object_id for key, call in calls.items()},
        )
        volume.commit()
        for key, call in calls.items():
            receipts[key] = call.get()
        publish('coordinator', status='running_comparison', finished=list(receipts))
    write(Path('/volume') / FID / 'completed-receipts.json', receipts)
    volume.commit()
    publish(
        'coordinator', status='fits_finished_pending_independent_closeout', finished=list(receipts)
    )
    return receipts


@app.local_entrypoint()
def main():
    call = coordinate.spawn()
    write(
        F / 'reproduce/coordinator-call.json',
        {
            'call_id': call.object_id,
            'family': FID,
            'submitted_unix': time.time(),
            'status': 'submitted',
        },
    )
    print('Saved durable coordinator:', call.object_id, flush=True)
