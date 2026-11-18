"""Pair-preserving source-context controls for inference sensitivity only."""
import copy
import hashlib
import json
import random
from collections import defaultdict

FIELDS = ('source_n', 'source_c', 'n_terminus', 'c_terminus')


def controlled(records, mode, seed=20261117):
    if mode not in ('original', 'unknown', 'within-study-shuffle'):
        raise ValueError(mode)
    pairs = defaultdict(dict)
    for row in records:
        assert row['split'] == 'validation' and row['value'] in (0, 1)
        pair = row['provenance']['pair_id']
        assert row['value'] not in pairs[pair]
        pairs[pair][row['value']] = row
    groups = defaultdict(list)
    for pair, members in pairs.items():
        assert set(members) == {0, 1}
        classes = {m.get('class_override') or m['mhc_class'] for row in members.values() for m in row['mhc']}
        studies = {str(row['provenance'].get('study_id') or row['provenance'].get('pmid')) for row in members.values()}
        lengths = {len(row['peptide']) for row in members.values()}
        assert len(classes) == len(studies) == len(lengths) == 1 and 'None' not in studies
        # Preserve each member's input availability. Jointly move both pair members,
        # including their terminus indicators, so role/length patterns stay comparable.
        knownness = tuple(members[label]['context'].get(k) is not None for label in (0, 1) for k in FIELDS)
        groups[(next(iter(classes)), next(iter(studies)), next(iter(lengths)), knownness)].append(pair)
    donors = {}
    for key, ids in sorted(groups.items()):
        ids = sorted(ids)
        shuffled = ids.copy()
        key_seed = int.from_bytes(hashlib.sha256(json.dumps([seed, key]).encode()).digest()[:8], 'big')
        random.Random(key_seed).shuffle(shuffled)
        donors.update(zip(ids, shuffled, strict=True))
    result, changed = [], 0
    for source in records:
        row = copy.deepcopy(source)
        pair, label = row['provenance']['pair_id'], row['value']
        donor = pairs[donors[pair]][label] if mode == 'within-study-shuffle' else source
        original = {k: source['context'].get(k) for k in FIELDS}
        replacement = {k: None if mode == 'unknown' else donor['context'].get(k) for k in FIELDS}
        changed += original != replacement
        row['context'].update(replacement)
        row['provenance']['flank_input_control'] = {
            'mode': mode, 'seed': seed, 'original_source_context': original,
            'donor_pair': donors[pair] if mode == 'within-study-shuffle' else None,
        }
        a, b = copy.deepcopy(source), copy.deepcopy(row)
        b['provenance'].pop('flank_input_control')
        for record in (a, b):
            for field in FIELDS:
                record['context'].pop(field, None)
        assert a == b
        result.append(row)
    return result, {
        'mode': mode, 'seed': seed, 'rows': len(records), 'pairs': len(pairs),
        'changed_rows': changed, 'groups': len(groups),
        'donor_mapping': donors if mode == 'within-study-shuffle' else None,
        'non_flank_invariants_verified': True,
        'scope': 'source-input corruption sensitivity, not a naturally occurring source or causal intervention',
    }
