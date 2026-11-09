"""Count canonical allele sets without duplicating measured outcomes per allele."""

from collections import Counter
from pathlib import Path
import hashlib
import json
import sys
import time


def summarize(manifest_path, output):
    manifest_path = Path(manifest_path)
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    counts, cells, populations = Counter(), Counter(), Counter()
    sources = {}
    started = time.time()
    for source in manifest['sources']:
        path = Path(source['path'])
        digest = hashlib.sha256()
        rows = 0
        with path.open('rb') as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                rows += 1
                split, endpoint = row['split'], row['endpoint']
                cls = row['mhc'][0]['mhc_class']
                generated = row['source'] == 'hitlist_ligand_background'
                populations[split, 'generated' if generated else 'original', endpoint, cls] += 1
                if generated:
                    continue
                counts[split, endpoint, cls, row['qualifier']] += 1
                provenance = row['provenance']
                alleles = sorted(
                    {
                        allele
                        for candidate in row['mhc']
                        for allele in candidate['provenance'].get('catalog_alleles', [])
                    }
                )
                dimensions = {
                    'catalog_allele_set': ';'.join(alleles) or 'unknown',
                    'catalog_candidate_set': json.dumps(
                        sorted(
                            sorted(candidate['provenance'].get('catalog_alleles', []))
                            for candidate in row['mhc']
                        ),
                        separators=(',', ':'),
                    ),
                    'candidate_count': str(len(row['mhc'])),
                    'study_component': row['group_id'],
                    'evaluation_era': provenance['evaluation_era'],
                    'label_kind': provenance.get('label_kind', 'physical_or_qualitative'),
                }
                for field, value in dimensions.items():
                    cells[split, endpoint, cls, field, value] += 1
                for allele in alleles:
                    cells[split, endpoint, cls, 'catalog_allele_membership', allele] += 1
        assert digest.hexdigest() == source['sha256'], 'source hash mismatch'
        sources[path.name] = {'sha256': digest.hexdigest(), 'rows': rows}
        print('Audited', path.name, rows, flush=True)
    result = {
        'status': 'counted',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'manifest_sha256': hashlib.sha256(raw_manifest).hexdigest(),
        'sources': sources,
        'counts': [
            dict(split=s, endpoint=e, mhc_class=c, qualifier=q, n=n)
            for (s, e, c, q), n in sorted(counts.items())
        ],
        'populations': [
            dict(split=s, population=p, endpoint=e, mhc_class=c, n=n)
            for (s, p, e, c), n in sorted(populations.items())
        ],
        'strata': [
            dict(split=s, endpoint=e, mhc_class=c, field=f, value=v, n=n)
            for (s, e, c, f, v), n in sorted(cells.items())
        ],
        'started_unix': started,
        'finished_unix': time.time(),
        'interpretation': 'Original observations counted once per exact candidate set; '
        'individual catalog allele memberships can overlap. No predictions '
        'or quantitative target values were read into the summaries.',
    }
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    summarize(sys.argv[1], sys.argv[2])
