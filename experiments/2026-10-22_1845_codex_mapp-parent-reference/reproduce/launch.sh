#!/bin/bash
set -euo pipefail
# Re-executes frozen recipes in a fresh registered directory under this repo.
# Original execution commands: python -u code/audit.py; python -u code/verify.py.
origin="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$origin/../.." && pwd)"
python_bin=/Users/iskander/code/shared-virtual-env/bin/python
cd "$repo"
"$python_bin" - "$origin" "${1:?Supply a fresh experiments/YYYY-MM-DD_HHMM_codex_slug directory}" <<'PYREPRO'
from pathlib import Path
import json,shutil,sys,subprocess
origin=Path(sys.argv[1]);repo=origin.parents[1];target=Path(sys.argv[2]).resolve()
assert target.parent==repo/'experiments', 'output must be directly under repository experiments/'
assert not target.exists(), 'refuse to overwrite any existing experiment'
for sub in ['code','reproduce/source','results','retrievals','launch_logs']:(target/sub).mkdir(parents=True,exist_ok=True)
for name in ['audit.py','verify.py']:shutil.copy2(origin/'reproduce/source'/name,target/'code'/name)
assets=json.loads((origin/'reproduce/reference_assets.json').read_text())
assets['seed_paths']=json.loads((origin/'results/input_paths.json').read_text())['references']
(target/'reproduce/reference_assets.json').write_text(json.dumps(assets,indent=2)+'\n')
shutil.copy2(origin/'reproduce/launch.sh',target/'reproduce/launch.sh')
(target/'README.md').write_text('Reproduction of '+str(origin.relative_to(repo))+'. Frozen mapping contracts unchanged; source audit only, no model or predictive split. Status: running.\n')
with (repo/'experiments/experiment_log.md').open('a') as stream:stream.write('\n## '+target.name+' — MAPP reference audit reproduction (registered)\n\nCodex / GPT-6 frozen recipe; see ['+target.name+']('+target.name+'/README.md). Original: '+origin.name+'. No training or predictive split.\n')
for name in ['audit.py','verify.py']:
 with (target/'launch_logs'/(name+'.log')).open('w') as log:subprocess.run([sys.executable,'-u',str(target/'code'/name)],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
print(target)
PYREPRO
