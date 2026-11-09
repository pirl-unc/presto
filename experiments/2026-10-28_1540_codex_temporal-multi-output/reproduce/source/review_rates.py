"""Read source identity and units for sparse kinetics; no sequence/model evaluation."""

from pathlib import Path
import hashlib
import json
import time
import modal

FID = '2026-10-28_1540_codex_temporal-multi-output'
app = modal.App('presto-temporal-rate-source-review')
volume = modal.Volume.from_name('presto-checkpoints')


@app.function(
    cpu=1,
    memory=4096,
    timeout=1800,
    nonpreemptible=True,
    volumes={'/volume': volume},
    max_containers=1,
)
def review():
    root = Path('/volume') / FID / 'prepared'
    result = []
    for split in ('train', 'validation', 'test'):
        with (root / ('hitlist-v4-iedb-' + split + '.jsonl')).open() as stream:
            for line in stream:
                row = json.loads(line)
                prov = row['provenance']
                if row['endpoint'] not in ('kon', 'koff') and not (
                    prov.get('study_id') == 'pmid:21257290' and row['endpoint'] == 'half_life'
                ):
                    continue
                raw = prov['raw_record']
                result.append(
                    dict(
                        id=row['observation_id'],
                        split=split,
                        endpoint=row['endpoint'],
                        value=row['value'],
                        unit=row['unit'],
                        qualifier=row['qualifier'],
                        study_id=prov['study_id'],
                        source={
                            k: raw.get(k)
                            for k in (
                                'assay_iri',
                                'reference_iri',
                                'reference_title',
                                'assay_comments',
                                'assay_method',
                                'response_measured',
                                'measurement_units',
                                'measurement_inequality',
                                'quantitative_measurement',
                                'quantitative_value',
                            )
                        },
                    )
                )
    return {'status': 'inspected_source_metadata', 'rows': result, 'finished_unix': time.time()}


@app.local_entrypoint()
def main():
    f = Path(__file__).resolve().parents[1]
    call_file = f / 'reproduce/rate-review-call.json'
    assert not call_file.exists(), 'use the saved source-review call'
    call = review.spawn()
    call_file.write_text(
        json.dumps(
            {
                'call_id': call.object_id,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'argv': ['modal', 'run', '--detach', str(Path(__file__).resolve())],
                'cpu': 1,
                'memory_mib': 4096,
                'nonpreemptible': True,
                'requested_gpu': None,
                'started_unix': time.time(),
            },
            indent=2,
        )
        + '\n'
    )
    result = call.get()
    (f / 'results/rate-training-source-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Inspected', len(result['rows']), 'kinetic-source records')
