"""Independent lightweight status polling; never waits on artifact archives."""

from pathlib import Path
import asyncio
import functools
import http.server
import json
import re
import threading
import time
import modal

F = Path(__file__).resolve().parents[1]
FID = F.name
D = modal.Dict.from_name('presto-cross-class-live')
server = http.server.ThreadingHTTPServer(
    ('127.0.0.1', 54502), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(F))
)
threading.Thread(target=server.serve_forever, daemon=True).start()


def read(p, default=None):
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return default


async def remote_snapshot():
    keys = [FID + ':preparation', FID + ':wide-multi-s42', FID + ':wide-multi-s43']
    return await asyncio.wait_for(asyncio.gather(*(D.get.aio(k) for k in keys)), timeout=30)


last_remote = [None, None, None]
remote_runner = asyncio.Runner()

while True:
    state = {
        'family': FID,
        'refreshed_unix': time.time(),
        'source_files': {},
        'runs': [],
        'preparation': None,
        'poll_error': None,
    }
    acquisition = read(F / 'results/acquisition.json', {})
    state['source_files'] = {
        k: {q: v[q] for q in ('size_bytes', 'sha256')}
        for k, v in acquisition.get('sources', {}).items()
    }
    dates = read(F / 'results/pubmed-2026-dates.json', {})
    state['verified_2026_studies'] = len(dates.get('records', []))
    try:
        p = max((F / 'launch_logs').glob('build-attempt*.log'), key=lambda p: p.stat().st_mtime)
        with p.open('rb') as stream:
            stream.seek(max(0, p.stat().st_size - 100000))
            tail = stream.read().decode(errors='replace')
        matches = re.findall(r'Scanning (iedb|cedar):\s*(\d+)%', tail)
        state['source_parse'] = {
            'iedb': 100 if any(k == 'cedar' for k, v in matches) else 0,
            'cedar': 0,
        }
        for name, percent in matches:
            state['source_parse'][name] = int(percent)
        if 'Published complete dataset:' in tail:
            state['source_parse'] = {'iedb': 100, 'cedar': 100}
    except FileNotFoundError:
        state['source_parse'] = {}
    try:
        last_remote = remote_runner.run(remote_snapshot())
    except Exception as exc:
        state['poll_error'] = repr(exc)
    state['preparation'] = last_remote[0]
    state['runs'] = [value for value in last_remote[1:] if value]
    state['admission'] = read(F / 'results/admission.json', {})
    prepared = read(F / 'results/prepared.json', {})
    if prepared.get('verification', {}).get('status') == 'passed':
        state['preparation'] = {
            'status': 'prepared',
            'stage': 'Frozen cohort passed canonical preflight and independent audit',
            'updated_unix': prepared['finished_unix'],
        }
    state['build'] = read(F / 'results/build.json', {})
    if (
        state['build'].get('status') == 'built'
        and state['preparation']
        and state['preparation'].get('status') == 'source_index_retry'
    ):
        state['preparation'] = {
            'status': 'admission_starting',
            'stage': 'Source build complete; starting canonical admission',
            'counts': {},
            'updated_unix': state['build']['updated_unix'],
        }
    path = F / 'results/live.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, allow_nan=False) + '\n')
    temp.replace(path)
    time.sleep(20)
