"""Compare a real v6 checkpoint under its frozen runtime and the v7 legacy reader."""

import json
import os
from pathlib import Path
import subprocess
import sys

F = Path(__file__).resolve().parents[1]
ROOT = F.parents[1]
OLD = ROOT / 'experiments/2026-10-29_2303_codex_assay-measurement-layer'
collection = json.loads((OLD / 'results/pooled-s42/collection.json').read_text())
run = Path(collection['directory'])
script = '''
import gzip,json,sys,torch
from pathlib import Path
from presto.training.modular import load_checkpoint
from presto.data.modular import Observation
run=Path(sys.argv[1])
rows={}
with gzip.open(run/'validation-predictions.jsonl.gz','rt') as stream:
    for line in stream:
        r=json.loads(line)
        if r['endpoint'] != 'ic50': continue
        method=r['assay_method']['detector']
        cls=r['observation']['mhc'][0]['mhc_class']
        rows.setdefault((cls,method), Observation.from_dict(r['observation']))
        if len(rows)==4: break
model,scales,payload=load_checkpoint(run/'raw/selected.pt')
model.eval()
with torch.no_grad(): outputs=model(list(rows.values()))
values = [dict(id=r.observation_id, prediction=float(p.value),
    panel={k:float(v) for k,v in p.assay_values.items()})
    for r,p in zip(rows.values(),outputs)]
print(json.dumps(values))
'''
frozen = Path(json.loads((OLD / 'reproduce/launch.json').read_text())['runtime_snapshot']).parent
outputs = []
for source in (frozen, ROOT.parent):
    env = {**os.environ, 'PYTHONPATH': str(source), 'OMP_NUM_THREADS': '1'}
    result = subprocess.run(
        [sys.executable, '-c', script, str(run)],
        cwd='/private/tmp',
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    outputs.append(json.loads(result.stdout))
assert outputs[0] == outputs[1], 'historical checkpoint predictions changed'
(F / 'results/historical-compatibility.json').write_text(
    json.dumps(
        {
            'status': 'bitwise_equal_cpu_predictions',
            'checkpoint_sha256': collection['checkpoint_sha256'],
            'cells': len(outputs[0]),
            'reference_runtime': str(frozen),
            'reader_runtime': str(ROOT.parent),
            'comparison': outputs[0],
        },
        indent=2,
    )
    + '\n'
)
print('Real v6 checkpoint: all four class/method cells and panel values match exactly.')
