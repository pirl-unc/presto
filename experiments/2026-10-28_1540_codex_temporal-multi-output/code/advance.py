"""Single-owner local continuation through verified preparation, fits and collection."""

from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time
import modal
from control import get_result

F = Path(__file__).resolve().parents[1]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    with (F / '.pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print('Waiting for the single queued preparation call', flush=True)
        while not (F / 'reproduce/prepare-call.json').exists():
            time.sleep(20)
        call_id = json.loads((F / 'reproduce/prepare-call.json').read_text())['call_id']
        dependency = modal.FunctionCall.from_id(call_id)
        result = get_result(dependency)
        assert result['status'] == 'prepared' and result['verification']['status'] == 'passed'
        write(F / 'results/prepared.json', result)
        write(F / 'results/partition-verification.json', result['verification'])
        for name, value in result['review_metadata'].items():
            write(F / 'results' / name, value)
        print('Prepared and independently verified; freezing exact fit budgets', flush=True)
        if not (F / 'reproduce/training-contract.json').exists():
            subprocess.run([sys.executable, str(F / 'code/freeze_contract.py')], check=True)
        process = None
        if not (F / 'reproduce/training-calls.json').exists():
            log = (F / 'launch_logs/training.log').open('a')
            process = subprocess.Popen(
                [sys.executable, '-m', 'modal', 'run', '--detach', str(F / 'code/launch.py')],
                env={**os.environ, 'PRESTO_MODAL_GPU': 'H100!'},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            while not (F / 'reproduce/training-calls.json').exists():
                if process.poll() is not None:
                    raise RuntimeError('GPU launcher stopped before publishing its call IDs')
                time.sleep(10)
        print(
            'Two frozen fit calls registered; collecting independently of status polling',
            flush=True,
        )
        with (F / 'launch_logs/collection.log').open('a') as log:
            subprocess.run(
                [sys.executable, str(F / 'analysis/collect.py')],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        with (F / 'results/prediction-verification.log').open('a') as log:
            subprocess.run(
                [sys.executable, str(F / 'analysis/verify_predictions.py')],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if process is not None:
            # Function receipts and hashes are authoritative if the original
            # launcher's local transport failed after the remote calls started.
            code = process.wait()
            write(F / 'results/launcher-exit.json', {'returncode': code})
        print('Fits collected and independently verified; scientific writeup remains', flush=True)


if __name__ == '__main__':
    main()
