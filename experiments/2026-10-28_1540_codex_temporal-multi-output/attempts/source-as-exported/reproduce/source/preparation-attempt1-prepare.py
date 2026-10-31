"""CPU-only materialization of the frozen prospective multi-output cohort."""

from pathlib import Path
import hashlib
import json
import sys
import time
import shutil
import socket
import modal

FID = '2026-10-28_1540_codex_temporal-multi-output'
app = modal.App('presto-temporal-partition-20261028')
volume = modal.Volume.from_name('presto-checkpoints')
live = modal.Dict.from_name('presto-cross-class-live')
REQUIREMENTS = [
    'hitlist==2.0.40',
    'torch==2.14.1',
    'mhcseqs==2.6.13',
    'numpy==2.5.4',
    'psutil==7.2.2',
]
image = (
    modal.Image.debian_slim(python_version='3.12')
    .uv_pip_install(*REQUIREMENTS)
    .env({'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
)
if modal.is_local():
    F = Path(__file__).resolve().parents[1]
    U = F.parents[1] / 'artifacts' / FID / 'upload'
    image = (
        image.add_local_dir(U / 'presto', remote_path='/runtime/presto')
        .add_local_file(U / 'runtime-pins.json', remote_path='/bundle/runtime-pins.json')
        .add_local_file(
            U / 'human-ensembl112.pep.fa.gz', remote_path='/bundle/human-ensembl112.pep.fa.gz'
        )
        .add_local_file(F / 'code/prepare_worker.py', remote_path='/bundle/prepare_worker.py')
        .add_local_file(F / 'code/partition.py', remote_path='/bundle/partition.py')
        .add_local_file(F / 'analysis/verify_cohort.py', remote_path='/bundle/verify_cohort.py')
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@app.function(
    image=image, cpu=8, memory=98304, timeout=14400, max_containers=1, volumes={'/runs': volume}
)
def run(admission_call_id):
    result = modal.FunctionCall.from_id(admission_call_id).get()
    assert result['status'] == 'admitted'
    volume.reload()
    sys.path.insert(0, '/bundle')
    from prepare_worker import prepare
    from verify_cohort import verify

    root = Path('/runs') / FID

    def status(stage, counts=None):
        live.put(
            FID + ':preparation',
            dict(status='partitioning', stage=stage, counts=counts or {}, updated_unix=time.time()),
        )
        print(stage, counts or {}, flush=True)

    try:
        result = prepare(root, status)
        result['prepared_record_sha256'] = sha(root / 'prepared/prepared.json')
        status('independent source, target, study, peptide and background verification')
        result['verification'] = verify(root)
        result['review_metadata'] = {
            name: json.loads((root / 'prepared' / name).read_text())
            for name in (
                'study-selection.json',
                'endpoint-support.json',
                'preflight.json',
                'wide-multi-s42.json',
                'wide-multi-s43.json',
            )
        }
        volume.commit()
        status('prepared; frozen manifests passed common-trainer preflight', result['counts'])
        return result
    except BaseException as exc:
        live.put(
            FID + ':preparation', dict(status='failed', stage=repr(exc), updated_unix=time.time())
        )
        volume.commit()
        raise


@app.local_entrypoint()
def main():
    call_id = json.loads((F / 'reproduce/admission-call.json').read_text())['call_id']
    if (F / 'reproduce/prepare-call.json').exists():
        raise ValueError('already attempted; recover by saved call ID')
    for path in (
        F / 'code/prepare.py',
        F / 'code/prepare_worker.py',
        F / 'code/partition.py',
        F / 'analysis/verify_cohort.py',
    ):
        shutil.copyfile(path, F / 'reproduce/source' / path.name)
    print('Waiting for frozen admission call', call_id, flush=True)
    dependency = modal.FunctionCall.from_id(call_id)
    while True:
        try:
            admitted = dependency.get(timeout=30)
            break
        except TimeoutError:
            continue
        except socket.gaierror as exc:
            print('Transient admission status lookup failed:', repr(exc), flush=True)
            time.sleep(5)
    assert admitted['status'] == 'admitted'
    write(F / 'results/admission.json', admitted)
    write(
        F / 'reproduce/prepare-launch.json',
        dict(
            argv=['modal', 'run', '--detach', str(Path(__file__).resolve())],
            sources={p.name: sha(p) for p in (F / 'code').glob('*.py')},
            dependency_call_id=call_id,
            requirements=REQUIREMENTS,
            requested_gpu=None,
            cpu=8,
            memory_mib=98304,
            started_unix=time.time(),
        ),
    )
    call = run.spawn(call_id)
    write(F / 'reproduce/prepare-call.json', {'call_id': call.object_id})
    result = call.get()
    write(F / 'results/prepared.json', result)
    write(F / 'results/partition-verification.json', result['verification'])
    for name, value in result['review_metadata'].items():
        write(F / 'results' / name, value)
    print('Prepared', result['counts'], flush=True)
