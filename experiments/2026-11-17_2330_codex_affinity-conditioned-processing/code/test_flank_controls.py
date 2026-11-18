"""Control transformations must preserve labels, MHC, APC and pair membership."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('processing_flank_controls', Path(__file__).with_name('flank_controls.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def records():
    return [
        {'split': 'validation', 'value': label, 'peptide': 'AAAA' if label else 'BBBB',
         'mhc': [{'mhc_class': 'II', 'groove1': 'C', 'groove2': 'D'}],
         'provenance': {'pair_id': str(pair), 'study_id': '123'},
         'context': {'source_n': 'A' * (pair + 1), 'source_c': 'C' * (pair + label + 1),
                     'n_terminus': False, 'c_terminus': False, 'cell_type': 'B-cell'}}
        for pair in range(12) for label in (0, 1)
    ]


def test_unknown_preserves_every_non_source_field():
    data = records()
    output, receipt = module.controlled(data, 'unknown')
    assert receipt['changed_rows'] == len(data)
    for original, row in zip(data, output, strict=True):
        assert all(row['context'][k] is None for k in module.FIELDS)
        assert row['context']['cell_type'] == original['context']['cell_type']
        assert row['peptide'] == original['peptide'] and row['mhc'] == original['mhc']
        assert row['provenance']['pair_id'] == original['provenance']['pair_id']
        assert original['context']['source_n'] is not None


def test_shuffle_moves_complete_pair_contexts_deterministically():
    data = records()
    output, receipt = module.controlled(data, 'within-study-shuffle')
    assert module.controlled(data, 'within-study-shuffle') == (output, receipt)
    assert receipt['changed_rows'] > 0
    lookup = {(r['provenance']['pair_id'], r['value']): r for r in data}
    for row in output:
        source = lookup[(receipt['donor_mapping'][row['provenance']['pair_id']], row['value'])]
        assert all(row['context'][k] == source['context'][k] for k in module.FIELDS)


def test_unknown_and_known_empty_flanks_never_mix_in_shuffle():
    data = records()
    for row in data[:2]:
        row['context']['source_n'] = None
        row['context']['n_terminus'] = None
    for row in data[2:4]:
        row['context']['source_n'] = ''
        row['context']['n_terminus'] = True
    output, _ = module.controlled(data, 'within-study-shuffle')
    for before, after in zip(data, output, strict=True):
        assert [(before['context'][k] is None) for k in module.FIELDS] == [(after['context'][k] is None) for k in module.FIELDS]
