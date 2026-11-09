"""Independent conservation, original targets, peptide embargoes and background audit."""

from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import sqlite3
import time

DERIVED = {
    'group_aliases',
    'study_component_years',
    'evaluation_era',
    'partition_policy',
    'evaluation_population',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(row):
    row = {k: v for k, v in row.items() if k not in ('split', 'group_id')}
    row['provenance'] = {k: v for k, v in row['provenance'].items() if k not in DERIVED}
    return hashlib.sha256(json.dumps(row, sort_keys=True, allow_nan=False).encode()).hexdigest()


def encoded(row):
    endpoint, value, unit = row['endpoint'], row['value'], row['unit']
    if endpoint in ('elution_observation', 'qualitative_binding'):
        return value
    if endpoint == 'td50':
        assert unit == 'C'
        return value
    conversions = {
        'pM': 0.001,
        'nM': 1.0,
        'uM': 1e3,
        'mM': 1e6,
        'M': 1e9,
        'min': 1.0,
        'h': 60.0,
        's': 1 / 60,
        '1/s': 1.0,
        '1/min': 1 / 60,
        '1/(M*s)': 1.0,
        'M^-1s^-1': 1.0,
        'nM^-1s^-1': 1e9,
    }
    return math.log10(value * conversions[unit])


def verify(root):
    start = time.monotonic()
    root = Path(root)
    p = root / 'prepared'
    receipt = json.loads((p / 'prepared.json').read_text())
    manifest = json.loads(Path(receipt['manifests']['wide-multi-s42']['path']).read_text())
    selections = json.loads((p / 'study-selection.json').read_text())
    splits = {a: c['split'] for c in selections['components'] for a in c['aliases']}
    groups = {a: c['id'] for c in selections['components'] for a in c['aliases']}
    db = sqlite3.connect(f'file:{root}/admission/admitted.sqlite?mode=ro', uri=True)
    priority = {'train': 0, 'validation': 1, 'test': 2}
    peptide_owner = {}
    for peptide, study in db.execute('SELECT peptide,study FROM rows WHERE excluded=0'):
        owner = splits[study]
        if priority[owner] > priority.get(peptide_owner.get(peptide), -1):
            peptide_owner[peptide] = owner
    originals = {}
    parents = {}
    excluded = Counter()
    for identifier, peptide, endpoint, study, payload in db.execute(
        'SELECT id,peptide,endpoint,study,data FROM rows WHERE excluded=0'
    ):
        if peptide_owner[peptide] != splits[study]:
            excluded['peptide_embargo'] += 1
            continue
        if endpoint not in manifest['model']['endpoints']:
            excluded['unsupported_output'] += 1
            continue
        row = json.loads(payload)
        originals[identifier] = (fingerprint(row), splits[study], groups[study])
        raw = row['provenance']['raw_record']
        if endpoint == 'elution_observation':
            if row['value'] == 1:
                assert raw['is_ms_observation'] is True and raw['ms_outcome_status'] in {
                    'positive_result',
                    'curated_ligand_table',
                }
                parents[identifier] = (
                    peptide,
                    json.dumps(row['mhc'], sort_keys=True),
                    json.dumps(row['context'], sort_keys=True),
                    splits[study],
                    groups[study],
                )
            else:
                assert (
                    raw['is_ms_observation'] is False
                    and raw['ms_outcome_status'] == 'negative'
                    and raw['qualitative_measurement'].lower().startswith('negative')
                )
        elif endpoint == 'qualitative_binding':
            label = raw['qualitative_measurement'].lower()
            assert row['value'] == (1.0 if label.startswith('positive') else 0.0)
            assert label.startswith(('positive', 'negative')) and row['qualifier'] == '='
        else:
            assert row['value'] == raw['quantitative_value']
            assert row['unit'] == (
                'C' if raw['measurement_units'] == '°C' else raw['measurement_units']
            )
            assert row['qualifier'] == (raw['measurement_inequality'] or '=')
    known_positive = {v for (v,) in db.execute('SELECT peptide FROM positive_peptides')}
    for left, right in db.execute(
        'SELECT r.study,s.study FROM rows r JOIN source_studies s '
        'ON r.provenance_id=s.provenance_id'
    ):
        assert splits[left] == splits[right]
    db.close()
    # Read exact pinned FASTA independently of the sampling/index implementation.
    proteins = {}
    name = None
    sequence = []
    with (p / 'human-ensembl112.pep.fa').open() as stream:
        for line in stream:
            if line.startswith('>'):
                if name is not None:
                    proteins[name] = ''.join(sequence)
                name = line[1:].split()[0]
                sequence = []
            else:
                sequence.append(line.strip())
        if name is not None:
            proteins[name] = ''.join(sequence)
    actual = set()
    peptides = {}
    aliases = {}
    counts = Counter()
    backgrounds = Counter()
    moments = {}
    for spec in manifest['sources']:
        path = Path(spec['path'])
        assert sha(path) == spec['sha256']
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                identifier = row['observation_id']
                prov = row['provenance']
                split = row['split']
                peptide = row['peptide']
                assert identifier not in actual
                actual.add(identifier)
                assert row['source'] == spec['source']
                assert peptides.setdefault(peptide, split) == split
                assert aliases.setdefault(row['group_id'], split) == split
                for alias in prov.get('group_aliases', []):
                    assert aliases.setdefault(alias, split) == split
                counts[
                    split
                    + ':'
                    + row['endpoint']
                    + ':'
                    + row['mhc'][0]['mhc_class']
                    + ':'
                    + row['qualifier']
                ] += 1
                if row['source'] == 'hitlist_ligand_background':
                    parent = prov['parent_observation_id']
                    meta = parents[parent]
                    assert row['value'] == 0 and row['unit'] == 'binary' and row['qualifier'] == '='
                    assert (
                        prov['measured_outcome'] is False
                        and prov['label_kind'] == 'generated_unobserved_background'
                    )
                    assert (
                        len(peptide) == len(meta[0])
                        and split == meta[3]
                        and row['group_id'] == meta[4]
                    )
                    assert (
                        json.dumps(row['mhc'], sort_keys=True) == meta[1]
                        and json.dumps(row['context'], sort_keys=True) == meta[2]
                    )
                    protein = proteins[prov['protein_id']]
                    pos = prov['protein_position']
                    assert (
                        protein[pos : pos + len(peptide)] == peptide
                        and peptide not in known_positive
                    )
                    if prov['background_kind'] == 'same_protein':
                        assert (
                            meta[0] in protein
                            and prov['protein_id'] in prov['positive_candidate_proteins']
                        )
                    else:
                        assert (
                            prov['background_kind'] == 'reference_proteome'
                            and not prov['positive_candidate_proteins']
                        )
                    token = (
                        f"{prov['policy_reference']}|{prov['seed']}|{parent}|{prov['replicate']}"
                    )
                    assert (
                        identifier
                        == 'ligand-background:' + hashlib.sha256(token.encode()).hexdigest()
                    )
                    backgrounds[parent] += 1
                else:
                    expected = originals.pop(identifier)
                    assert (fingerprint(row), split, row['group_id']) == expected, identifier
                if (
                    split == 'train'
                    and row['qualifier'] == '='
                    and row['endpoint'] not in ('elution_observation', 'qualitative_binding')
                ):
                    key = row['endpoint']
                    value = encoded(row)
                    n, mean, m2 = moments.get(key, (0, 0.0, 0.0))
                    n += 1
                    delta = value - mean
                    mean += delta / n
                    m2 += delta * (value - mean)
                    moments[key] = (n, mean, m2)
    assert not originals
    assert counts == Counter(receipt['counts'])
    import gzip

    exhausted = {}
    with gzip.open(p / 'partition-dispositions.jsonl.gz', 'rt') as stream:
        for line in stream:
            note = json.loads(line)
            if note['reason'] == 'background_exhausted':
                exhausted[note['id']] = note['replicate']
    for parent, meta in parents.items():
        expected = 1 if meta[3] == 'train' else 5
        assert backgrounds[parent] == exhausted.get(parent, expected)
    scales = json.loads((p / 'preflight.json').read_text())['scales']
    for endpoint, (n, mean, m2) in moments.items():
        scale = scales[endpoint]
        assert n == scale['exact_count']
        assert math.isclose(mean, scale['mean'], rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(
            max(0.001, math.sqrt(m2 / n)), scale['scale'], rel_tol=1e-9, abs_tol=1e-9
        )
    result = dict(
        status='passed',
        original_measurements=len(actual) - sum(backgrounds.values()),
        generated_comparators=sum(backgrounds.values()),
        independent_exclusions=dict(excluded),
        source_and_target_conservation=True,
        whole_study_aliases_disjoint=True,
        exact_peptides_disjoint=True,
        background_sequence_lineage_and_positive_blacklist_verified=True,
        background_ratios_verified=True,
        training_scales_independently_verified=True,
        seconds=time.monotonic() - start,
    )
    (p / 'independent-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
