"""Project every training support row without consulting measured quantities."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

FAMILY = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def project():
    cfg = json.loads((FAMILY / 'reproduce/launch.json').read_text())
    for path, expected in cfg['source_hashes'].items():
        assert sha(path) == expected, path
    recipe = Path(cfg['recipe_directory'])
    manifest = json.loads((recipe / 'manifest.json').read_text())
    assert sha(recipe / 'examples.json') == manifest['files']['examples.json']['sha256']
    examples = json.loads((recipe / 'examples.json').read_text())
    counts = Counter()
    grouped = defaultdict(list)
    for e in examples:
        # Do not access observations, targets, reference signals or source intensities.
        counts[e['split']] += len(e['common_positive_times'])
        if e['split'] != 'train':
            continue
        grouped[e['condition_id']].append(e)
    assert counts['train'] == 90630 and counts['val'] == 14822
    parents = Counter(v[0]['construct_id'] for v in grouped.values())
    blocks = []
    for condition, curves in sorted(grouped.items()):
        first = curves[0]
        sequence = first['parent_sequence']
        context = first['context']
        enzyme = context['substrate_concentration_um'] / context['enzyme_substrate_denominator']
        by_time = defaultdict(list)
        for e in curves:
            assert e['parent_sequence'] == sequence and e['context'] == context
            assert e['construct_id'] == first['construct_id'] and e['split'] == 'train'
            a, b = e['start_0based'], e['end_0based']
            # The canonical recipe already admits one source span per curve.
            assert 0 <= a < b <= len(sequence)
            assert sequence[a:b] == e['peptide_sequence']
            for t in e['common_positive_times']:
                assert t > 0
                by_time[t].append(dict(curve_id=e['curve_id'], start=a, end=b,
                    time=t, weight=1/(len(parents)*parents[e['construct_id']]*len(curves)*len(e['common_positive_times']))))
        for t, rows in sorted(by_time.items()):
            spans = sorted({(r['start'], r['end']) for r in rows})
            blocks.append(dict(block_id=len(blocks), condition=condition,
                construct_id=first['construct_id'], sequence=sequence,
                context_log_enzyme=math.log(enzyme/0.2), time_hours=t,
                spans=spans, rows=rows))
    assert sum(len(b['rows']) for b in blocks) == 90630
    assert abs(sum(r['weight'] for b in blocks for r in b['rows'])-1) < 1e-10
    initial = json.loads(Path(cfg['initial_fit']).read_text())['sequence_context']['coefficients']
    assert len(initial) == 40
    result = dict(initial_coefficients=initial, blocks=blocks, counts=dict(counts),
        source_hashes=cfg['source_hashes'], measured_quantities_accessed=False,
        source_serialization_contains_all_splits=True)
    dump(FAMILY/'reproduce/training-support.json', result)
    summary = dict(n_blocks=len(blocks), n_conditions=len(grouped), n_parents=len(parents),
        n_measurements=90630, n_required_span_times=sum(len(b['spans']) for b in blocks),
        lengths=sorted({len(b['sequence']) for b in blocks}), times=sorted({b['time_hours'] for b in blocks}),
        support_sha256=sha(FAMILY/'reproduce/training-support.json'), quantities_accessed=False)
    dump(FAMILY/'results/projection-summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    project()
