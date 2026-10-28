"""Receipt-gated, hashed collection; never blocks the independent status dashboard."""

from pathlib import Path
import hashlib
import json
import shutil
import tarfile
import time
import modal

F = Path(__file__).resolve().parents[1]
ROOT = F.parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def collect(condition, receipt):
    assert receipt['condition'] == condition
    storage = ROOT / 'artifacts' / F.name / 'collected'
    storage.mkdir(parents=True, exist_ok=True)
    archive = storage / (condition + '.tar.gz')
    if not archive.exists():
        temporary = archive.with_suffix('.partial')
        with temporary.open('wb') as stream:
            for chunk in modal.Volume.from_name('presto-checkpoints').read_file(receipt['archive']):
                stream.write(chunk)
        assert temporary.stat().st_size == receipt['bytes']
        assert sha(temporary) == receipt['sha256']
        temporary.replace(archive)
    assert archive.stat().st_size == receipt['bytes'] and sha(archive) == receipt['sha256']
    destination = storage / condition
    if not destination.exists():
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                assert parts and parts[0] == condition and '..' not in parts
                assert not member.name.startswith('/') and (member.isfile() or member.isdir())
            tar.extractall(storage, filter='data')
    report = json.loads((destination / 'report.json').read_text())
    assert sha(destination / 'raw/selected.pt') == report['checkpoint_sha256']
    review = F / 'results' / condition
    review.mkdir(exist_ok=True)
    # Small summaries are canonical; checkpoints and all per-example dumps stay at
    # their explicitly recorded large-artifact locations.
    for name in ('report.json', 'hardware.json', 'progress.json', 'status.json'):
        shutil.copyfile(destination / name, review / name)
    for name in ('resolved-config.json', 'history.json', 'admission.json'):
        shutil.copyfile(destination / 'raw' / name, review / name)
    result = dict(
        status='collected',
        condition=condition,
        receipt=receipt,
        directory=str(destination.resolve()),
        predictions={
            split: dict(
                path=str((destination / f'{split}-predictions.jsonl.gz').resolve()),
                sha256=sha(destination / f'{split}-predictions.jsonl.gz'),
            )
            for split in ('train', 'validation', 'test')
        },
        collected_unix=time.time(),
    )
    write(review / 'collection.json', result)
    return result


def main():
    calls = json.loads((F / 'reproduce/training-calls.json').read_text())
    pending = dict(calls)
    while pending:
        for name, identifier in list(pending.items()):
            try:
                receipt = modal.FunctionCall.from_id(identifier).get(timeout=0)
            except TimeoutError:
                continue
            write(F / 'results' / (name + '-receipt.json'), receipt)
            result = collect(name, receipt)
            print(name, result['status'], flush=True)
            del pending[name]
        if pending:
            time.sleep(30)
    # Choose a seed exclusively from validation fields before opening any test dump.
    scores = {}
    hashes = {}
    for name in calls:
        path = F / 'results' / name / 'report.json'
        report = json.loads(path.read_text())
        losses = report['summary']['metrics']['validation']
        scores[name] = sum(v['loss'] for v in losses.values()) / len(losses)
        hashes[name] = sha(path)
    write(
        F / 'results/validation-decision.json',
        dict(
            status='frozen',
            winner=min(scores, key=lambda n: (scores[n], n)),
            criterion='minimum equal-endpoint validation loss',
            scores=scores,
            report_sha256=hashes,
            test_used_for_selection=False,
        ),
    )


if __name__ == '__main__':
    main()
