"""CPU-only audit of immutable existing source files; no model execution."""

from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tarfile
import time

import modal

FID = '2026-10-29_1428_codex_assay-methods-censoring'
PARENT = '2026-10-28_1540_codex_temporal-multi-output'
app = modal.App('presto-assay-methods-censoring-20261029')
volume = modal.Volume.from_name('presto-checkpoints')
image = modal.Image.debian_slim(python_version='3.12').uv_pip_install(
    'numpy==2.5.4', 'scipy==1.18.1', 'pyarrow==26.0.0'
)
if modal.is_local():
    F = Path(__file__).resolve().parents[1]
    image = image.add_local_file(F / 'analysis/audit_methods.py', '/bundle/audit_methods.py')
    image = image.add_local_file(F.parent / PARENT / 'results/build.json', '/bundle/build.json')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@app.function(
    image=image, cpu=4, memory=16384, timeout=3600, max_containers=1, volumes={'/volume': volume}
)
def run():
    sys.path.insert(0, '/bundle')
    from audit_methods import train_audit, source_census

    root = Path('/volume') / PARENT
    output = Path('/volume') / FID / 'audit'
    assert not output.exists(), 'Preserve the original attempt'
    summary = train_audit(
        root / 'revisions/kinetic-review-v2/prepared/hitlist-v4-iedb-train.jsonl',
        'c76aeafbb47c66b7fa904c191502de95f8812d45275cdfe680821838a37200b1',
        output,
    )
    build = json.loads(Path('/bundle/build.json').read_text())
    specs = {
        name: {**spec, 'path': str(Path('/volume') / Path(spec['path']).relative_to('/runs'))}
        for name, spec in build['files'].items()
        if name in ('binding.parquet', 'other_assays.parquet')
    }
    summary['source_census'] = source_census(specs, output)
    summary['environment'] = {
        n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'pyarrow')
    }
    summary['python'] = sys.version
    summary['requested_gpu'] = None
    summary['cpu'] = 4
    summary['memory_mib'] = 16384
    summary['worker_sha256'] = sha('/bundle/audit_methods.py')
    summary['build_receipt_sha256'] = sha('/bundle/build.json')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    archive = output.parent / 'audit.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(output, arcname='audit')
    volume.commit()
    return dict(
        status='completed',
        archive=str(archive.relative_to('/volume')),
        bytes=archive.stat().st_size,
        sha256=sha(archive),
        counts=summary['counts'],
        seconds=summary['seconds'],
        census=summary['source_census'],
    )


@app.local_entrypoint()
def main():
    f = Path(__file__).resolve().parents[1]
    for name in ('reproduce/source', 'results', 'launch_logs'):
        (f / name).mkdir(parents=True, exist_ok=True)
    call_path = f / 'reproduce/call.json'
    assert not call_path.exists()
    for path in (Path(__file__), f / 'analysis/audit_methods.py'):
        shutil.copyfile(path, f / 'reproduce/source' / path.name)
    launch = dict(
        argv=['python', '-m', 'modal', 'run', '--detach', __file__],
        agent_model='Codex / GPT-6',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_dirty=subprocess.check_output(['git', 'status', '--short'], text=True),
        started_unix=time.time(),
        requested_gpu=None,
        cpu=4,
        memory_mib=16384,
        source_hashes={p.name: sha(p) for p in (f / 'reproduce/source').glob('*.py')},
    )
    (f / 'reproduce/launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    call = run.spawn()
    call_path.write_text(json.dumps(dict(call_id=call.object_id), indent=2) + '\n')
    receipt = call.get()
    (f / 'results/receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
