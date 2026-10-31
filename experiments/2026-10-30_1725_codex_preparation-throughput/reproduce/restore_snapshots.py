"""Rebuild the exact recorded runtimes from Git and the five frozen overlays."""

import hashlib
import json
from pathlib import Path
import subprocess


def main():
    bundle = Path(__file__).resolve().parent
    contract = json.loads((bundle / 'launch.json').read_text())
    repository = bundle.parents[2]
    for variant, runtime in contract['runtimes'].items():
        for relative, expected in runtime['sha256'].items():
            frozen = bundle / 'source' / variant / relative
            content = (
                frozen.read_bytes()
                if frozen.exists()
                else subprocess.check_output(
                    ['git', 'show', contract['git_commit'] + ':' + relative], cwd=repository
                )
            )
            assert hashlib.sha256(content).hexdigest() == expected, (variant, relative)
            destination = Path(runtime['path']) / relative
            if destination.exists():
                assert destination.read_bytes() == content, ('refusing overwrite', destination)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
    print('Both 219-file runtime snapshots match the recorded hashes.')


if __name__ == '__main__':
    main()
