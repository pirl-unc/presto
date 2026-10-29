"""Subtract only the verified source-review rows from historical descriptive counts."""

from collections import Counter
from pathlib import Path
import hashlib
import json

F = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = F / 'attempts/source-as-exported/results/cohort-strata.json'
    dispositions = F / 'results/source-review-dispositions.json'
    verified = read(F / 'results/partition-verification.json')
    assert verified['status'] == 'passed' and verified['removed_rows'] == 20
    cells = Counter(
        {(r['split'], r['endpoint'], r['field'], r['value']): r['n'] for r in read(original)}
    )
    rows = read(dispositions)
    assert len(rows) == len({r['observation_id'] for r in rows}) == 20
    for record in rows:
        row = record['original']
        raw = row['provenance']['raw_record']
        assert row['split'] == 'train' and row['endpoint'] in ('koff', 'kon')
        fields = {
            field: str(raw.get(field) or 'unknown')
            for field in ('assay_method', 'mhc_species', 'source_organism', 'host', 'species')
        }
        fields.update(peptide_length=str(len(row['peptide'])), candidate_count=str(len(row['mhc'])))
        for field, value in fields.items():
            cells[row['split'], row['endpoint'], field, value] -= 1
        for mhc in row['mhc']:
            for allele in mhc['provenance'].get('reported_candidates', []):
                cells[row['split'], row['endpoint'], 'reported_candidate_allele', allele] -= 1
    assert min(cells.values()) >= 0
    result = [
        dict(split=s, endpoint=e, field=f, value=v, n=n)
        for (s, e, f, v), n in sorted(cells.items())
        if n
    ]
    prepared = read(F / 'results/prepared.json')
    totals = Counter()
    for item in result:
        if item['field'] == 'candidate_count':
            totals[item['split'] + ':' + item['endpoint']] += item['n']
    assert dict(totals) == prepared['physical_observations']
    output = F / 'results/cohort-strata.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    receipt = dict(
        status='passed',
        original_sha256=sha(original),
        dispositions_sha256=sha(dispositions),
        result_sha256=sha(output),
        source_sha256=sha(Path(__file__)),
        removed_rows=20,
        checked='Verified row-removal delta and every original split/endpoint total',
    )
    (F / 'results/cohort-strata-revision.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(receipt)


if __name__ == '__main__':
    main()
