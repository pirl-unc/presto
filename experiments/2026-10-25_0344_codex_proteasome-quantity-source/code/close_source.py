"""Close source evidence with independent support counts and native run mapping."""

import collections
import csv
import gzip
import hashlib
import importlib.metadata
import json
import resource
import time
import xml.etree.ElementTree as ET
from pathlib import Path, PureWindowsPath

EXP = Path(__file__).resolve().parents[1]
CONFIG = json.loads((EXP / 'reproduce/launch.json').read_text())
RAW = Path(CONFIG['raw_directory'])
start = time.monotonic()


def read(name):
    return json.loads((EXP / name).read_text())


verified = read('results/independent-recovery-verification.json')
assert verified['passed']
support = read('results/panel-support.json')
inventory = read('results/workbook-inventory.json')
cells = Path(inventory['independent_original_cells'])
assert hashlib.sha256(cells.read_bytes()).hexdigest() == inventory['independent_original_cells_sha256']
samples = support['sample_labels']
columns = dict(zip(('AB', 'AC', 'AD', 'AE', 'AF', 'AG'), samples))
counts = {s: collections.Counter() for s in samples}
sequences = set()
seen = set()
with gzip.open(cells, 'rt') as stream:
    for line in stream:
        r = json.loads(line)
        if r['file'] != '41467_2021_26427_MOESM10_ESM.xlsx':
            continue
        coordinate = r['cell']
        col = coordinate.rstrip('0123456789')
        number = int(coordinate[len(col):])
        if number == 1:
            continue
        if col == 'A':
            sequences.add(r['value'])
        if col in columns:
            assert (col, number) not in seen
            seen.add((col, number))
            v = r['value']
            assert isinstance(v, (int, float)) and v >= 0
            counts[columns[col]]['positive' if v > 0 else 'zero'] += 1
assert len(sequences) == support['unique_sequences'] == support['data_rows']
assert len(seen) == 6 * support['data_rows']
assert {k: dict(v) for k, v in counts.items()} == support['measurement_state_counts']

metadata_name = 'PXD018722-project-files-page0-retry1.json'
assert json.loads((RAW / (metadata_name + '.receipt.json')).read_text())['access_status'] == 'retrieved'
files = json.loads((RAW / metadata_name).read_text())
assert json.loads((RAW / 'PXD018722-project-files-page1.json').read_text()) == []
assert len({r['accession'] for r in files}) == len(files)
for name in ('Hi20S_mqpar.xml', 'Hi20S_experimentalDesignTemplate.txt'):
    r = next(r for r in files if r['fileName'] == name)
    payload = (RAW / name).read_bytes()
    assert len(payload) == r['fileSizeBytes']
    assert hashlib.sha1(payload).hexdigest() == r['checksum']
design = list(csv.DictReader((RAW / 'Hi20S_experimentalDesignTemplate.txt').open(), delimiter='\t'))
mq = ET.fromstring((RAW / 'Hi20S_mqpar.xml').read_bytes())
paths = [PureWindowsPath(x.text).name for x in mq.findall('filePaths/string')]
labels = [x.text for x in mq.findall('experiments/string')]
assert len(paths) == len(labels) == len(design) == len(set(paths)) == 6
mapping = []
for d, path, label in zip(design, paths, labels):
    assert d['Name'] + '.raw' == path and d['Experiment'] == label
    assert d['Fraction'] == '' and d['PTM'] == ''
    native = next(r for r in files if r['fileName'] == path)
    assert native['fileCategory']['value'] == 'RAW'
    mapping.append({'sample_label': label, 'raw_filename': path,
                    'raw_accession': native['accession'], 'raw_downloaded': False,
                    'design_fraction': d['Fraction'], 'design_ptm': d['PTM']})
assert sorted(labels) == sorted(samples)
keys = ['maxQuantVersion', 'matchBetweenRuns', 'matchBetweenRunsFdr', 'peptideFdr',
        'proteinFdr', 'minPeptideLengthForUnspecificSearch', 'maxPeptideLengthForUnspecificSearch',
        'parameterGroups/parameterGroup/enzymeMode', 'parameterGroups/parameterGroup/lfqMode',
        'parameterGroups/parameterGroup/lfqSkipNorm']
search = {k: mq.findtext(k) for k in keys}
search['variable_modifications'] = [x.text for x in mq.findall('parameterGroups/parameterGroup/variableModifications/string')]
search['fixed_modifications'] = [x.text for x in mq.findall('parameterGroups/parameterGroup/fixedModifications/string')]
search['fasta_basename'] = PureWindowsPath(mq.findtext('fastaFiles/FastaFileInfo/fastaFilePath')).name
assert search['minPeptideLengthForUnspecificSearch'] == '8'
assert search['maxPeptideLengthForUnspecificSearch'] == '25'
assert search['matchBetweenRuns'] == 'True'
context = {'project': 'PXD018722', 'source_panel': 'Sahu2021_Dataset7_Hi20S',
           'run_mapping': mapping, 'search_settings': search,
           'article_minimum_length_claim': 7, 'deposited_search_length_range': [8, 25],
           'context': 'Human HEK293T control versus inducible PSMD2/Rpn1 knockdown; native intracellular hot-water peptide extraction described by primary study.',
           'not_assumed': ['professional_APC', 'complete_KO', 'numeric_remaining_activity',
                           'matched_precursor_quantity', 'known_extraction_recovery', 'molar_calibration',
                           'independent_biological_replicates_from_numbered_labels',
                           'paired_control_KD_specimens', 'Intensity_is_LFQ', 'unmodified_peptidoform'],
           'original_file_inventory_rows': len(files), 'original_raw_files_downloaded': 0}
(EXP / 'results/run-search-context.json').write_text(json.dumps(context, indent=2) + '\n')
receipts = [json.loads(p.read_text()) for p in sorted(RAW.glob('*.receipt.json'))]
assets = [{'name': p.name, 'path': str(p), 'size_bytes': p.stat().st_size,
           'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
          for p in sorted(RAW.rglob('*')) if p.is_file()]
(EXP / 'results/source-assets.json').write_text(json.dumps({'assets': assets, 'retrievals': receipts}, indent=2) + '\n')
with (EXP / 'results/panel-support.csv').open('w', newline='') as stream:
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['source_sample', 'positive_signal_cells', 'source_zero_cells'])
    for sample in samples:
        writer.writerow([sample, counts[sample]['positive'], counts[sample]['zero']])
result = {'passed': True, 'source_fidelity_with_explicit_exceptions': True,
          'source_peptide_rows': support['data_rows'],
          'positive_signal_cells': sum(x['positive'] for x in counts.values()),
          'source_zero_cells': sum(x['zero'] for x in counts.values()),
          'run_maps_verified': 6, 'original_file_inventory_rows': len(files),
          'validated_apc_concentration_model': False, 'training_rows_admitted': 0,
          'heldout_predictions': None, 'test_absence_reason': 'Source qualification precedes shared canonical admission and prospective modeling.',
          'synthetic_labels': False, 'pretraining': None, 'training': None,
          'next_action': 'Admit native PSMD2-knockdown source and source-supported context through Hitlist; verify original reference and sequence-level quantitative aggregation before a frozen biological comparison.',
          'dryad_status': 'Metadata verified; public file403 and API401 prevent numerical admission.',
          'seconds': time.monotonic() - start,
          'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'packages': {n: importlib.metadata.version(n) for n in ('openpyxl', 'numpy')}}
with (EXP / 'results/source-closure.json').open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result))
