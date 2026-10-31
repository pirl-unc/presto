"""Frozen two-seed H100 fits; preparation and all scientific membership are upstream."""

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import modal

FID = '2026-10-29_1956_codex_curated-data-retrain'
REVISION = 'source-review-v2'
GPU = os.environ.get('PRESTO_MODAL_GPU', 'H100!')
if GPU != 'H100!':
    raise ValueError('Frozen hardware contract requires H100!')
CONDITIONS = ('curated-multi-s42', 'curated-multi-s43')
REQUIREMENTS = [
    'torch==2.14.1',
    'numpy==2.5.4',
    'pyyaml==6.0.3',
    'mhcgnomes==3.64.6',
    'mhcseqs==2.6.13',
    'tqdm==4.70.1',
    'psutil==7.2.2',
]
app = modal.App('presto-curated-data-retrain-20261029')
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
        }
    )
)
if modal.is_local():
    F = Path(__file__).resolve().parents[1]
    U = F.parents[1] / 'artifacts' / FID / 'upload'
    image = (
        image.add_local_dir(U / 'presto', remote_path='/runtime/presto')
        .add_local_file(U / 'runtime-pins.json', remote_path='/bundle/runtime-pins.json')
        .add_local_file(
            F / 'reproduce/training-contract.json', remote_path='/bundle/training-contract.json'
        )
        .add_local_file(F / 'code/worker.py', remote_path='/bundle/worker.py')
        .add_local_file(
            F / 'code/checkpoint_observer.py', remote_path='/bundle/checkpoint_observer.py'
        )
    )


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    path = Path(path)
    p = path.with_suffix(path.suffix + '.tmp')
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    p.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def snapshot(path, condition):
    report = read(path / 'report.json', {})
    history = read(path / 'raw/history.json', [])
    latest = history[-1].get('validation', {}) if history else {}
    evaluated = read(path / 'live-evaluation.json', {})
    return dict(
        condition=condition,
        status=read(path / 'status.json', {'status': 'starting'}),
        progress=read(path / 'progress.json', {}),
        history=history[-20:],
        hardware=read(path / 'hardware.json', {}),
        metrics={
            'validation': report.get('summary', {}).get('metrics', {}).get('validation', latest)
        },
        ligand_validation=evaluated.get('ligands', {}),
        selected_steps=report.get('selected_steps'),
        updated_unix=time.time(),
    )


def publish(key, value):
    try:
        live.put(key, value)
    except Exception as exc:
        print('Status publication failed; work continues:', repr(exc), flush=True)


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=65536,
    ephemeral_disk=524288,
    timeout=86400,
    startup_timeout=1200,
    retries=0,
    max_containers=2,
    single_use_containers=True,
    volumes={'/volume': volume},
)
def fit(condition):
    assert condition in CONDITIONS
    contract = read('/bundle/training-contract.json')
    assert contract['status'] == 'frozen'
    assert contract['source_review']['revision'] == REVISION
    assert sha('/bundle/runtime-pins.json') == contract['runtime_pins_sha256']
    durable = Path('/volume') / FID / 'revisions' / REVISION
    root = Path('/runs') / FID
    assert sha(durable / 'prepared/prepared.json') == contract['prepared_record_sha256']
    destination = durable / condition
    destination.mkdir(parents=True, exist_ok=True)
    receipt = read(destination / 'artifact-receipt.json')
    if receipt:
        archive = Path('/volume') / receipt['archive']
        assert sha(archive) == receipt['sha256'] and archive.stat().st_size == receipt['bytes']
        return receipt
    if (destination / 'report.json').exists():
        return package(destination, condition)
    resume = (destination / 'raw/resume.pt').exists()
    if not resume and (destination / 'status.json').exists():
        assert read(destination / 'progress.json', {}).get('steps', 0) == 0, (
            'training progressed but its durable snapshot is missing'
        )
        if (destination / 'raw').exists():
            (destination / 'raw').rename(destination / ('initialization-' + str(time.time_ns())))
        (destination / 'status.json').rename(
            destination / ('initialization-status-' + str(time.time_ns()) + '.json')
        )
    staged = time.monotonic()
    publish(
        FID + ':' + condition,
        dict(
            condition=condition,
            status={'status': 'staging_verified_inputs'},
            updated_unix=time.time(),
        ),
    )
    (root / 'prepared').mkdir(parents=True)
    shutil.copyfile(durable / 'prepared/prepared.json', root / 'prepared/prepared.json')
    spec = read(root / 'prepared/prepared.json')['manifests'][condition]
    manifest_path = Path(spec['path'])
    assert manifest_path.is_relative_to(root / 'prepared')
    shutil.copyfile(durable / manifest_path.relative_to(root), manifest_path)
    assert sha(manifest_path) == spec['sha256']
    staged_bytes = 0
    for source in read(manifest_path)['sources']:
        path = Path(source['path'])
        assert path.is_relative_to(root / 'prepared')
        shutil.copyfile(durable / path.relative_to(root), path)
        assert sha(path) == source['sha256']
        staged_bytes += path.stat().st_size
    write(
        destination / 'storage.json',
        dict(
            policy='hash-verified local inputs; unchanged canonical manifest paths',
            staged_bytes=staged_bytes,
            staging_seconds=time.monotonic() - staged,
            requested_ephemeral_disk_mib=524288,
        ),
    )
    process = subprocess.Popen(
        [sys.executable, '-u', '/bundle/worker.py', condition, str(destination)]
        + (['--resume'] if resume else []),
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    )
    key = FID + ':' + condition
    last_commit = time.monotonic()
    while process.poll() is None:
        publish(key, snapshot(destination, condition))
        if time.monotonic() - last_commit >= 300:
            try:
                volume.commit()
            except Exception as exc:
                print('Periodic commit failed; worker continues:', repr(exc), flush=True)
            last_commit = time.monotonic()
        time.sleep(10)
    value = snapshot(destination, condition)
    if process.returncode:
        value['status'] = dict(status='failed', error='worker exit ' + str(process.returncode))
        publish(key, value)
        volume.commit()
        raise RuntimeError(value['status']['error'])
    return package(destination, condition)


def package(destination, condition):
    report = read(destination / 'report.json')
    assert report['condition'] == condition and report['serving_replay']['status'] == 'passed'
    assert sha(destination / 'raw/selected.pt') == report['checkpoint_sha256']
    key = FID + ':' + condition
    value = snapshot(destination, condition)
    value['status'] = {'status': 'packaging'}
    publish(key, value)
    archive = destination.parent / (condition + '-review.tar.gz')
    # Full raw run remains on the volume; compact review bundle preserves every
    # held-out record, selected weights and replay without duplicating plain dumps.
    with tarfile.open(archive, 'w:gz') as tar:
        for p in sorted(destination.iterdir()):
            if p.is_file():
                tar.add(p, arcname=condition + '/' + p.name)
        for p in sorted((destination / 'raw').iterdir()):
            if p.suffix == '.json' or p.name in ('selected.pt', 'initial.pt', 'last.pt'):
                tar.add(p, arcname=condition + '/raw/' + p.name)
    receipt = dict(
        condition=condition,
        archive=str(archive.relative_to('/volume')),
        sha256=sha(archive),
        bytes=archive.stat().st_size,
        raw_directory=str(destination),
    )
    write(destination / 'artifact-receipt.json', receipt)
    volume.commit()
    value.update(status={'status': 'completed'}, artifact=receipt)
    publish(key, value)
    return receipt


@app.local_entrypoint()
def main():
    sys.path.insert(0, str(F / 'code'))
    from control import get_result

    contract = read(F / 'reproduce/training-contract.json')
    assert contract['status'] == 'frozen'
    assert read(F / 'results/launch-readiness.json', {}).get('status') == 'passed', (
        'checkpoint recovery integration has not passed its prelaunch checks'
    )
    readiness = read(F / 'results/launch-readiness.json')
    assert readiness['training_contract_sha256'] == sha(F / 'reproduce/training-contract.json')
    assert readiness['partition_verification_sha256'] == sha(
        F / 'results/partition-verification.json'
    )
    for relative, digest in readiness['source_hashes'].items():
        assert sha(F / relative) == digest
    assert sha(U / 'runtime-pins.json') == contract['runtime_pins_sha256']
    assert not (F / 'reproduce/training-calls.json').exists(), 'already launched; use saved calls'
    assert not (F / 'reproduce/training-call-progress.json').exists(), (
        'partial submission exists; recover saved calls without resubmitting'
    )
    assert read(F / 'results/partition-verification.json')['status'] == 'passed'
    invocation = dict(
        argv=['modal', 'run', '--detach', str(Path(__file__).resolve())],
        environment={'PRESTO_MODAL_GPU': GPU},
        requirements=REQUIREMENTS,
        requested_gpu=GPU,
        max_concurrent_gpus=2,
        ephemeral_disk_mib=524288,
        storage_policy='hash-verified local inputs; unchanged canonical manifest paths',
        conditions=CONDITIONS,
        contract=contract,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_dirty=subprocess.check_output(['git', 'status', '--short'], text=True),
        sources={p.name: sha(p) for p in (F / 'code').glob('*.py')},
        started_unix=time.time(),
    )
    for path in (F / 'code').glob('*.py'):
        shutil.copyfile(path, F / 'reproduce/source' / path.name)
    placeholder = F / 'reproduce/launch.json'
    if placeholder.exists():
        placeholder.rename(F / 'reproduce/planned-launch.json')
    write(F / 'reproduce/launch.json', invocation)
    (F / 'reproduce/launch.sh').write_text(
        '#!/bin/sh\nset -eu\nexport PRESTO_MODAL_GPU="H100!"\nexec modal run --detach "'
        + str(Path(__file__).resolve())
        + '" "$@"\n'
    )
    calls = {}
    for name in CONDITIONS:
        calls[name] = fit.spawn(name)
        write(
            F / 'reproduce/training-call-progress.json',
            dict(
                status='submitting',
                calls={key: call.object_id for key, call in calls.items()},
                pending=[key for key in CONDITIONS if key not in calls],
                training_contract_sha256=sha(F / 'reproduce/training-contract.json'),
                worker_sha256=sha(F / 'code/worker.py'),
            ),
        )
    write(
        F / 'reproduce/training-calls.json', {name: call.object_id for name, call in calls.items()}
    )
    # Independent status polling remains live even if collection or this connection fails.
    for name, call in calls.items():
        result = get_result(call)
        write(F / 'results' / (name + '-receipt.json'), result)
        print(name, 'finished', result['bytes'], 'review bytes', flush=True)
