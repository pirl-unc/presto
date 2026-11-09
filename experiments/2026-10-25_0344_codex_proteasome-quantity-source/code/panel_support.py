"""Describe source schema and measurement support without fitting or effect summaries."""

import collections
import hashlib
import json
import math
import resource
import shutil
import time
from pathlib import Path

from openpyxl import load_workbook

EXP = Path(__file__).resolve().parents[1]
RAW = Path(json.loads((EXP / 'reproduce/launch.json').read_text())['raw_directory'])
assert shutil.disk_usage(RAW).free >= 1024**3 + 32 * 1024**2
assert json.loads((EXP / 'results/independent-recovery-verification.json').read_text())['passed']
start = time.monotonic()
path = RAW / 'complete-members/41467_2021_26427_MOESM10_ESM.xlsx'
wb = load_workbook(path, read_only=True, data_only=True)
sheet = wb['peptides']
it = sheet.iter_rows(values_only=True)
headers = next(it)
assert len(headers) == len(set(headers)) == 35
samples = [f'{group}{i}' for group in ('Cont', 'KD') for i in range(1, 4)]
counts = {s: collections.Counter() for s in samples}
identification = {s: collections.Counter() for s in samples}
flags = {key: collections.Counter() for key in ('Reverse', 'Potential contaminant')}
seqs, proteins = set(), set()
patterns = collections.Counter()
peptide_lengths = []
n = 0
for values in it:
    n += 1
    row = dict(zip(headers, values))
    seqs.add(row['Sequence'])
    peptide_lengths.append(len(row['Sequence']))
    if row['Proteins']:
        proteins.update(row['Proteins'].split(';'))
    mask = []
    for sample in samples:
        value = row[f'Intensity {sample}']
        if value is None:
            state = 'missing'
        elif not isinstance(value, (int, float)) or not math.isfinite(value):
            state = 'nonfinite_or_nonnumeric'
        else:
            state = 'positive' if value > 0 else 'zero' if value == 0 else 'negative'
        counts[sample][state] += 1
        mask.append(state == 'positive')
        identification[sample][str(row[f'Identification type {sample}'])] += 1
    patterns[''.join('1' if x else '0' for x in mask)] += 1
    for key in flags:
        flags[key][str(row[key])] += 1
wb.close()
assert n == 10647
result = {'source_file': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
          'sheet': 'peptides', 'data_rows': n, 'unique_sequences': len(seqs),
          'literal_protein_accession_tokens': len(proteins), 'headers': headers,
          'sample_labels': samples, 'measurement_state_counts': counts,
          'identification_type_counts': identification, 'source_flags': flags,
          'positive_presence_patterns': patterns,
          'observed_sequence_length_range': [min(peptide_lengths), max(peptide_lengths)],
          'amount_or_effect_summaries_computed': False, 'training_rows_admitted': 0,
          'zero_interpretation': 'Source zero; cause of nondetection unknown, not a true peptide absence.',
          'quantity_contract': 'Source sequence-level Intensity columns; not certified LFQ values, unmodified-peptidoform amounts, precursor-normalized abundance, or molarity.',
          'scope': 'Source support only; no condition-based eligibility or model scores.',
          'seconds': time.monotonic() - start,
          'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
with (EXP / 'results/panel-support.json').open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps({k: v for k, v in result.items() if k not in
                  ('headers', 'positive_presence_patterns', 'source_flags', 'identification_type_counts')}))
