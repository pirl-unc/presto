"""Read compact durable run state without spawning another GPU worker."""
import argparse
import json
from datetime import datetime, timezone

import modal
from common import NAME, E, dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', required=True)
    args = parser.parse_args()
    volume = modal.Volume.from_name('presto-checkpoints')
    result = {'time': datetime.now(timezone.utc).isoformat(), 'runs': {}}
    for run in args.runs.split(','):
        current = {}
        for file in ('receipt.json', 'failure.json', 'hardware.json', 'raw/history.json'):
            try:
                value = json.loads(b''.join(volume.read_file(NAME + '/fits/' + run + '/' + file)))
            except FileNotFoundError:
                continue
            current[file] = value[-1:] if file.endswith('history.json') else value
        result['runs'][run] = current
    dump(E / 'results/latest-fit-progress.json', result)
    compact = {}
    for run, files in result['runs'].items():
        history = files.get('raw/history.json') or []
        compact[run] = {
            'closed': 'receipt.json' in files,
            'failure': files.get('failure.json'),
            'last_validated_step': history[-1]['steps'] if history else None,
            'coverage': files.get('receipt.json', {}).get('coverage'),
        }
    print(json.dumps({'time': result['time'], 'runs': compact}))


if __name__ == '__main__':
    main()
