"""Frozen CPU source audit, independent of the two active H100 fits."""

from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
import modal

FID = '2026-10-29_1318_codex_affinity-objectives-geometry'
PARENT = '2026-10-28_1540_codex_temporal-multi-output'
app = modal.App('presto-affinity-assay-agreement-20261029')
volume = modal.Volume.from_name('presto-checkpoints')
image = modal.Image.debian_slim(python_version='3.12').uv_pip_install(
    'numpy==2.5.4', 'scipy==1.18.1'
)
if modal.is_local():
    F = Path(__file__).resolve().parents[1]
    MANIFEST = F.parent / PARENT / 'reproduce/manifests/wide-multi-s42.json'
    image = image.add_local_file(
        F / 'analysis/assay_agreement.py', '/bundle/assay_agreement.py'
    ).add_local_file(MANIFEST, '/bundle/manifest.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


@app.function(
    image=image, cpu=4, memory=16384, timeout=3600, max_containers=1, volumes={'/volume': volume}
)
def run_audit():
    import importlib.metadata

    sys.path.insert(0, '/bundle')
    from assay_agreement import audit

    manifest = json.loads(Path('/bundle/manifest.json').read_text())
    assert (
        sha('/bundle/manifest.json')
        == '5541024b53ce7d09ef6779639eb9fb1b7fd2f72c770f84ba311a5b294053b9a1'
    )
    root = Path('/volume') / PARENT / 'revisions/kinetic-review-v2/prepared'
    sources = [
        (root / Path(s['path']).name, s['sha256'])
        for s in manifest['sources']
        if Path(s['path']).name == 'hitlist-v4-iedb-train.jsonl'
    ]
    assert len(sources) == 1
    output = Path('/volume') / FID / 'assay-agreement'
    assert not output.exists(), 'Preserve the original audit; use a declared attempt directory'
    result = audit(sources, output)
    result['execution'] = dict(
        requested_gpu=None,
        cpu=4,
        memory_mib=16384,
        python=sys.version,
        environment={name: importlib.metadata.version(name) for name in ('numpy', 'scipy')},
        manifest_sha256=sha('/bundle/manifest.json'),
        worker_sha256=sha('/bundle/assay_agreement.py'),
    )
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    archive = output.parent / 'assay-agreement.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(output, arcname=output.name)
    volume.commit()
    return dict(
        status='completed_source_audit',
        archive=str(archive.relative_to('/volume')),
        bytes=archive.stat().st_size,
        sha256=sha(archive),
        summary=result,
    )


@app.local_entrypoint()
def main():
    f = Path(__file__).resolve().parents[1]
    for name in ('results', 'reproduce/source', 'launch_logs'):
        (f / name).mkdir(parents=True, exist_ok=True)
    call_path = f / 'reproduce/audit-call.json'
    assert not call_path.exists(), 'Resume collection from the saved call; do not duplicate'
    for path in (Path(__file__), f / 'analysis/assay_agreement.py'):
        shutil.copyfile(path, f / 'reproduce/source' / path.name)
    launch = dict(
        status='launched',
        agent_model='Codex / GPT-6',
        argv=['python', '-m', 'modal', 'run', '--detach', str(Path(__file__).resolve())],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_dirty=subprocess.check_output(['git', 'status', '--short'], text=True),
        environment={'requested_gpu': None},
        started_unix=time.time(),
        source_hashes={p.name: sha(p) for p in (f / 'reproduce/source').glob('*.py')},
    )
    (f / 'reproduce/launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    call = run_audit.spawn()
    call_path.write_text(
        json.dumps(dict(call_id=call.object_id, started_unix=time.time()), indent=2) + '\n'
    )
    receipt = call.get()
    (f / 'results/audit-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(
        json.dumps(
            {
                'status': receipt['status'],
                'counts': receipt['summary']['counts'],
                'seconds': receipt['summary']['seconds'],
            }
        ),
        flush=True,
    )


if __name__ == '__main__':
    raise SystemExit('Launch with python -m modal run --detach code/audit.py')
