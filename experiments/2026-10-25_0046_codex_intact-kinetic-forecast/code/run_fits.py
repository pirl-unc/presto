"""Sequential guarded fits; stop on failure, never restart a prior attempt."""

import json
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
config=json.loads((root/'reproduce/launch.json').read_text())
for stage in ['controls',*config['fit_conditions']]:
    subprocess.run([sys.executable,str(root/'code/launch.py'),stage],check=True)
