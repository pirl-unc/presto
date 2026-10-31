"""Verify late source candidates and audit frozen comparison labels without edits."""

from collections import Counter
from pathlib import Path
import csv
import gzip
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import time

import modal

FID = '2026-10-29_1653_codex_measurement-curation-recovery'
FIT = '2026-10-29_1956_codex_curated-data-retrain'
PARENT = '2026-10-28_1540_codex_temporal-multi-output'
TAG = 'late-candidate-audit'
app = modal.App('presto-late-source-candidate-audit-20261029')
volume = modal.Volume.from_name('presto-checkpoints')
image = modal.Image.debian_slim(python_version='3.12')
if modal.is_local():
    F = Path(__file__).resolve().parents[1]
    A = Path('/Users/iskander/code/presto/artifacts') / FID
    image = image.add_local_file(
        A / 'recovery-batch-late/recovered-late/strazar_2023_recovered_unmodified.csv',
        '/bundle/source.csv',
    )
    image = image.add_local_file(
        F / 'results/recovery-admission-late-v2.json', '/bundle/admission.json'
    )
    image = image.add_local_file(F.parent / FIT / 'results/prepared.json', '/bundle/prepared.json')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(row):
    return row['peptide'], tuple(sorted((m['groove1'], m['groove2']) for m in row['mhc']))


def verify_local():
    receipt = json.loads((F / 'results/recovery-admission-late-v2.json').read_text())
    summary = receipt['summary']
    assert sha(A / Path(receipt['archive']).name) == receipt['sha256']
    root = A / 'recovery-batch-late/recovered-admission-late-v2'
    for name, expected in summary['files'].items():
        path = root / name
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']
    source = A / 'recovery-batch-late/recovered-late/strazar_2023_recovered_unmodified.csv'
    assert sha(source) == summary['source_sha256']
    with source.open() as stream:
        raw = {r['source_evidence_id']: r for r in csv.DictReader(stream)}
    identifiers, identities, evidence_ids = set(), set(), set()
    with gzip.open(root / 'train.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            p = row['provenance']
            review = p['peptidoform_review']
            r = raw[review['evidence_id']]
            assert row['observation_id'] not in identifiers
            assert identity(row) not in identities
            identifiers.add(row['observation_id'])
            identities.add(identity(row))
            evidence_ids.add(review['evidence_id'])
            assert r['peptide'] == row['peptide'] and len(row['peptide']) >= 9
            assert row['endpoint'] == 'elution_observation' and row['value'] == 1
            assert row['split'] == 'train' and row['group_id'] == 'pmid:37301199'
            assert len(row['mhc']) == 1 and row['mhc'][0]['mhc_class'] == 'II'
            assert p['source_sha256'] == summary['source_sha256']
            assert review['evidence_ledger_sha256'] == summary['source_evidence_sha256']
            assert p['raw_record']['has_ptm'] is False
            assert p['raw_record']['mhc_restriction'] == r['mhc_restriction']
            assert p['study_id'] == 'pmid:37301199'
            assert p['group_aliases'] == ['pmid:37301199']
            assert p['study_component_years'] == [2023]
    assert len(identifiers) == len(evidence_ids) == summary['train_observations']
    short = Counter(len(r['peptide']) for r in raw.values() if len(r['peptide']) < 9)
    assert sum(short.values()) == summary['counts']['excluded:unsupported_peptide']
    return dict(
        status='passed',
        archive_sha256=receipt['sha256'],
        verified_candidate_observations=len(identifiers),
        short_source_observations_by_length=dict(short),
        heldout_exclusions=summary['counts']['heldout_peptide_embargo'],
        source_sha256=summary['source_sha256'],
        member_hashes_verified=summary['files'],
    )


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=3600,
    max_containers=1,
    volumes={'/volume': volume},
)
def run():
    started = time.monotonic()
    summary = json.loads(Path('/bundle/admission.json').read_text())['summary']
    source = Path('/bundle/source.csv')
    assert sha(source) == summary['source_sha256']
    with source.open() as stream:
        reviewed = {r['peptide'] for r in csv.DictReader(stream)}
    candidate = Path('/volume') / FID / 'recovered-admission-late-v2/train.jsonl.gz'
    assert sha(candidate) == summary['files']['train.jsonl.gz']['sha256']
    with gzip.open(candidate, 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    identities = {identity(r) for r in rows}
    peptides = {r['peptide'] for r in rows}
    assert len(rows) == len(identities) == summary['train_observations']
    db = sqlite3.connect('file:/volume/' + PARENT + '/admission/admitted.sqlite?mode=ro', uri=True)
    missing = {
        peptide
        for peptide in reviewed
        if db.execute('SELECT 1 FROM positive_peptides WHERE peptide=?', (peptide,)).fetchone()
        is None
    }
    db.close()
    base = Path('/volume') / FIT / 'revisions/source-review-v2/prepared'
    prepared = json.loads(Path('/bundle/prepared.json').read_text())
    manifest_path = base / 'curated-multi-s42.json'
    assert sha(manifest_path) == prepared['manifests']['curated-multi-s42']['sha256']
    manifest = json.loads(manifest_path.read_text())
    intersections, heldout, hashes, current_positives = [], set(), {}, set()
    for spec in manifest['sources']:
        path = base / Path(spec['path']).name
        is_background = spec['source'] == 'hitlist_ligand_background'
        is_heldout = path.name.endswith(('-validation.jsonl', '-test.jsonl'))
        is_recovered = 'recovered' in path.name and not is_background
        if not (is_background or is_heldout or is_recovered):
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                if is_heldout:
                    heldout.add(row['peptide'])
                if is_recovered and row['value'] == 1 and row['split'] == 'train':
                    assert row['endpoint'] == 'elution_observation'
                    current_positives.add(identity(row))
                if row['peptide'] in reviewed and (is_background or is_heldout):
                    intersections.append(
                        dict(
                            observation_id=row['observation_id'],
                            split=row['split'],
                            generated=is_background,
                            peptide_sha256=hashlib.sha256(row['peptide'].encode()).hexdigest(),
                            candidate_eligible=row['peptide'] in peptides,
                            previously_blacklisted=row['peptide'] not in missing,
                        )
                    )
        assert digest.hexdigest() == spec['sha256']
        hashes[path.name] = digest.hexdigest()
    assert current_positives and current_positives <= identities
    assert not peptides & heldout
    return dict(
        status='passed',
        current_fit_unchanged=True,
        requested_gpu=None,
        current_manifest_sha256=sha(manifest_path),
        reviewed_unique_sequences=len(reviewed),
        candidate_observations=len(identities),
        current_recovered_positives_retained=len(current_positives),
        additional_candidate_pmhcs=len(identities - current_positives),
        missing_reviewed_from_original_blacklist=len(missing),
        candidates_with_heldout_peptides=0,
        intersection_counts=dict(
            Counter(
                r['split'] + (':generated' if r['generated'] else ':original')
                for r in intersections
            )
        ),
        intersections=intersections,
        scanned_source_hashes=hashes,
        seconds=time.monotonic() - started,
        interpretation=(
            'Source-verified presentation makes generated comparisons suspect; it does '
            'not establish binding to every comparator MHC. These candidates require '
            'a new frozen background/cohort preparation. Active fit inputs and '
            'prespecified sensitivity denominators are unchanged.'
        ),
    )


@app.local_entrypoint()
def main():
    receipt = F / 'results' / (TAG + '.json')
    assert not receipt.exists()
    local = verify_local()
    shutil.copyfile(__file__, F / 'reproduce/source' / (TAG + '.py'))
    call = run.spawn()
    launch = dict(
        call_id=call.object_id,
        argv=sys.argv,
        source_sha256=sha(__file__),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        dirty_status=subprocess.check_output(['git', 'status', '--porcelain'], text=True),
        started_unix=time.time(),
        requested_gpu=None,
        cpu=4,
        memory_mib=8192,
        local_verification=local,
    )
    (F / 'reproduce' / (TAG + '-launch.json')).write_text(json.dumps(launch, indent=2) + '\n')
    result = dict(local_verification=local, background_audit=call.get())
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(
        json.dumps(
            {
                k: v
                for k, v in result['background_audit'].items()
                if k not in ('intersections', 'scanned_source_hashes')
            }
        ),
        flush=True,
    )


if __name__ == '__main__':
    raise SystemExit('Launch with python -m modal run --detach <this file>')
