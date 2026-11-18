"""Link the new conditional TRAIN population to its actual APC annotations."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import modal
from common import E, NAME, BROAD, BASE, VOLUME, rows, sha, dump

app = modal.App('presto-processing-context-census')
image = BASE
if modal.is_local():
    image = image.add_local_dir(E / 'code', '/bundle').add_local_dir(E / 'reproduce/runtime', '/runtime/presto')


@app.function(image=image, cpu=4, memory=8192, timeout=3600, volumes={'/volume': VOLUME})
def audit():
    from project_context import Projection
    root = Path('/volume') / NAME
    prepared = json.loads((root / 'processing-preparation.json').read_text())
    source = next(r for r in prepared['sources'] if r['path'].endswith('processing-backgrounds-train.jsonl'))
    assert sha(source['path']) == source['sha256']
    config = json.loads((Path('/volume') / BROAD / 'fits/expanded-state-sqrt-s42-n6000/manifest.json').read_text())['model']
    projection = Projection(config)
    totals, cells, counts = Counter(), Counter(), Counter()
    studies, conditions = defaultdict(set), defaultdict(set)
    for raw in rows(source['path']):
        assert raw['split'] == 'train' and raw['value'] == 0 and raw['provenance']['processing_pair_positive_id']
        record = projection.record(raw)[0]
        classes = {m.get('class_override') or m['mhc_class'] for m in record['mhc']}
        assert len(classes) == 1
        cls = next(iter(classes))
        context, provenance = record['context'], record['provenance']
        totals[(cls, 'pairs')] += 1
        for field in ('source_n', 'source_c', 'cell_type'):
            totals[(cls, field + '_known')] += context.get(field) is not None
        totals[(cls, 'quantitative_component_annotations')] += bool(context.get('components') or context.get('component_overrides'))
        cell = context.get('cell_type_override') or context.get('cell_type') or '<unknown>'
        cells[(cls, str(cell))] += 1
        for field in ('cytokines', 'interventions', 'processing_pathways'):
            for name, value in context.get(field, {}).items():
                state = value.get('state', '<unspecified>') if isinstance(value, dict) else str(value)
                key = (cls, field, name, state)
                counts[key] += 1
                studies[key].add(str(provenance.get('study_id') or provenance.get('pmid')))
                if provenance.get('condition_id'):
                    conditions[key].add(str(provenance['condition_id']))
    result = {
        'source': source, 'test_accessed': False, 'projection': projection.receipt(),
        'totals': {cls: {key: n for (c, key), n in totals.items() if c == cls} for cls in ('I', 'II')},
        'cell_types': [{'class': cls, 'cell_type': cell, 'pairs': n} for (cls, cell), n in sorted(cells.items())],
        'annotations': [{'class': cls, 'field': field, 'name': name, 'state': state, 'pairs': n,
                         'studies': sorted(studies[key]), 'conditions': len(conditions[key])}
                        for key, n in sorted(counts.items()) for cls, field, name, state in [key]],
        'interpretation': 'One row per matched peptide pair. Counts reflect admitted annotations after the inherited model vocabulary projection, not isolated intervention/control comparisons; concurrent factors overlap.',
    }
    assert result['totals']['I']['pairs'] == prepared['pairs']['I']
    assert result['totals']['II']['pairs'] == prepared['pairs']['II']
    dump(root / 'processing-context-census.json', result)
    VOLUME.commit()
    return result


@app.local_entrypoint()
def main():
    call = audit.spawn()
    dump(E / 'reproduce/context-census-call.json', {'call': call.object_id})
    result = call.get()
    dump(E / 'results/processing-context-census.json', result)
    print(json.dumps({'totals': result['totals'], 'annotation_cells': len(result['annotations'])}))
