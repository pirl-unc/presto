"""Read-only CPU provenance follow-up to the frozen assay-agreement audit."""

from collections import Counter, defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import time

import modal

FID = '2026-10-29_1318_codex_affinity-objectives-geometry'
PARENT = '2026-10-28_1540_codex_temporal-multi-output'
app = modal.App('presto-assay-overlap-lineage-20261029')
volume = modal.Volume.from_name('presto-checkpoints')
image = modal.Image.debian_slim(python_version='3.12')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@app.function(image=image, cpu=4, memory=8192, timeout=3600, volumes={'/volume': volume})
def run():
    start = time.monotonic()
    root = Path('/volume') / FID
    summary = json.loads((root / 'assay-agreement/summary.json').read_text())
    source = summary['sources'][0]
    wanted = set()
    with gzip.open(root / 'assay-agreement/paired-values.jsonl.gz', 'rt') as stream:
        for line in stream:
            wanted.add(json.loads(line)['pmhc_sha256'])
    output = root / 'overlap-lineage'
    assert not output.exists()
    output.mkdir()
    groups = defaultdict(Counter)
    h = hashlib.sha256()
    kept = 0
    with (
        Path(source['path']).open('rb') as stream,
        gzip.open(output / 'overlap-records.jsonl.gz', 'wt') as out,
    ):
        for line in stream:
            h.update(line)
            row = json.loads(line)
            if row['endpoint'] not in ('ic50', 'kd', 'ec50', 'kd_via_ic50', 'kd_via_ec50'):
                continue
            p = row['provenance']
            raw = p['raw_record']
            fields = (
                'reference_iri',
                'pmid',
                'submission_id',
                'reference_title',
                'assay_method',
                'response_measured',
                'assay_comments',
            )
            metadata = dict(
                study_id=p['study_id'],
                group_aliases=p['group_aliases'],
                endpoint=row['endpoint'],
                **{k: raw.get(k) for k in fields},
            )
            groups[row['group_id']][json.dumps(metadata, sort_keys=True)] += 1
            mhc = row['mhc'][0]
            cls = mhc.get('class_override') or mhc['mhc_class']
            pid = hashlib.sha256(
                json.dumps(
                    (row['peptide'], mhc['groove1'], mhc['groove2'], cls), sort_keys=True
                ).encode()
            ).hexdigest()
            if pid not in wanted:
                continue
            record = dict(
                pmhc_sha256=pid,
                mhc_class=cls,
                endpoint=row['endpoint'],
                observation_id=row['observation_id'],
                group_id=row['group_id'],
                assay_id=p['original_measurement_id'],
                value=row['value'],
                unit=row['unit'],
                qualifier=row['qualifier'],
            )
            out.write(json.dumps(record) + '\n')
            kept += 1
    assert h.hexdigest() == source['sha256']
    details = [
        dict(group_id=g, metadata=json.loads(m), n=n)
        for g, counts in sorted(groups.items())
        for m, n in counts.most_common()
    ]
    (output / 'study-details.json').write_text(json.dumps(details, indent=2) + '\n')
    receipt = dict(
        status='completed_lineage_census',
        source=source,
        overlap_records=kept,
        study_components=len(groups),
        requested_gpu=None,
        seconds=time.monotonic() - start,
        validation_test_used=False,
        model_fitted=False,
    )
    (output / 'summary.json').write_text(json.dumps(receipt, indent=2) + '\n')
    archive = root / 'overlap-lineage.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(output, arcname=output.name)
    volume.commit()
    return dict(
        **receipt,
        archive=str(archive.relative_to('/volume')),
        bytes=archive.stat().st_size,
        sha256=sha(archive),
    )


@app.local_entrypoint()
def main():
    f = Path(__file__).resolve().parents[1]
    path = f / 'reproduce/lineage-call.json'
    assert not path.exists()
    shutil.copyfile(__file__, f / 'reproduce/source/lineage.py')
    launch = dict(
        argv=['python', '-m', 'modal', 'run', '--detach', __file__],
        source_sha256=sha(__file__),
        requested_gpu=None,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_dirty=subprocess.check_output(['git', 'status', '--short'], text=True),
        started_unix=time.time(),
    )
    (f / 'reproduce/lineage-launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    call = run.spawn()
    path.write_text(json.dumps(dict(call_id=call.object_id), indent=2) + '\n')
    receipt = call.get()
    (f / 'results/lineage-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
