from pathlib import Path
import hashlib
import json

import presto
import torch
from presto.models import Presto, PrestoConfig
from presto.models.modular import ModularPresto
from presto.data.modular import AA, MHCInput, Observation, fit_target_scales
from presto.training import train_step, save_checkpoint, load_checkpoint
from presto.inference import ObservationPredictor
from presto.scripts.train_remote import main
from presto.cli.train import main as canonical_main


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(230)
root = Path('/private/tmp/presto-unified-wheel-smoke')
assert Path(presto.__file__).is_relative_to(root)
assert ModularPresto is Presto
assert main is canonical_main
rows = [
    Observation(
        observation_id=f'fixture:{kind}',
        source='software-only',
        endpoint='ic50',
        peptide=AA[:15],
        mhc=(MHCInput(AA, AA[::-1], kind),),
        value=100.0 if kind == 'I' else 10.0,
        unit='nM',
        group_id='fixture',
    )
    for kind in ('I', 'II')
]
model = Presto(PrestoConfig(dropout=0))
scales = fit_target_scales(rows)
train_step(model, torch.optim.AdamW(model.parameters(), lr=0.001), rows, scales)
checkpoint = root / 'shared-smoke.pt'
save_checkpoint(checkpoint, model, scales)
restored, _, payload = load_checkpoint(checkpoint)
assert type(restored) is Presto and payload['format'] == 'presto.shared.v3'
outputs = ObservationPredictor(checkpoint).predict(rows)
assert len(outputs) == 2
for row, expected in zip(outputs, (1, 7)):
    registers = row['diagnostics']['candidates'][0]['registers'][0]
    assert len(registers['starts']) == expected
wheel = Path('/private/tmp/presto-unified-distributions/presto-0.1.22-py3-none-any.whl')
report = {
    'status': 'passed',
    'source': str(presto.__file__),
    'format': payload['format'],
    'fixture_rows': len(rows),
    'register_counts': [1, 7],
    'remote_launcher_is_canonical': True,
    'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
}
report_path = Path(
    '/private/tmp/presto-unified-core-20261027/presto/tasks/unified_training_core_20261027_wheel.json'
)
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
