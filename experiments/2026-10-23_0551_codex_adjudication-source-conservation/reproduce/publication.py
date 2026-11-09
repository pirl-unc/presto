import base64
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path('/private/tmp/presto-modular-model')
FAMILY = ROOT / 'experiments/2026-10-23_0551_codex_adjudication-source-conservation'
REPORT = 'https://github.com/pirl-unc/presto/blob/82feaf2/experiments/2026-10-23_0551_codex_adjudication-source-conservation/README.md'


def api(endpoint, payload=None):
    command = ['gh', 'api', endpoint]
    if payload is None:
        return json.loads(subprocess.check_output(command, text=True))
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as stream:
        json.dump(payload, stream)
        stream.flush()
        return json.loads(subprocess.check_output(command + ['--method', 'PATCH', '--input', stream.name], text=True))


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


section = f'''## Verified combined source generation — October 23, 2026

[Full report and immutable reproduction]({REPORT}). Executed Hitlist `0219827`
merges main 2.0.4 without dropping its nine multimer reviews. New external
successors use schema9/catalog12 and accept verified reference or adjudication
predecessors; both historical schema8 families remain readable, and ambiguous
envelopes are rejected. This addresses the integration defect in #771.

The actual combined package retains all126 predecessor files, all33 MAPP and
23 calibration files, and all9 review claims. Independent hashing/direct Parquet
replay checks183files and1,853,840mixed-grain rows across32canonical query
relations. Allfour existing recipe payloads, contributors and partitions remain
byte-identical. Source provenance changes to the new generation. The initial
standalone/canonical recipe-envelope baseline error and amended verification
are retained; no producer or target was changed/rerun to obtain a pass.

Normal local `./test.sh` passes3,317tests/10warnings in321.51s with the unchanged
memoryguard andoneworker. Format/lint/70focused checks/strict docs pass. Producer
148.01s/3.28GiBpeakRSS; independent verifier12.83s/0.64GiB, local CPU only.
Code is on draft767; 2.0.5 remains unreleased. HostedCI/review/merge/release are
separate gates. Controls admission remains in #759; no new biological fit or
concentration claim is made here.
'''
updates = [
    ('repos/pirl-unc/hitlist/issues/771', section, None),
    ('repos/pirl-unc/hitlist/pulls/767', section, 'Add MS calibration evidence and preserve assay reviews (2.0.5)'),
    ('repos/pirl-unc/presto/pulls/131', f'''## Reviewed-evidence and source conservation — October 23, 2026

[Registered conservation report]({REPORT}); [Hitlist issue771](https://github.com/pirl-unc/hitlist/issues/771).
The shared schema9 package now preserves all9 multimer reviews together with
MAPP and calibration data. Independent replay verifies183files,32queries and
allfour unchanged recipe outputs. Hitlist's normal local suite passes3,317tests;
the fix is pushed to draft767. No new biological fit: availability/precursor/MS
limits and the earlier MAPP uniform-baseline failure remain open. Next is shared
admission of the recovered cellular controls under Hitlist759.
''', None),
]
receipts = []
for endpoint, addition, title in updates:
    old = api(endpoint)
    assert old['state'] == 'open', endpoint
    if '/pulls/' in endpoint:
        assert old['draft'], endpoint
    marker = addition.splitlines()[0]
    assert marker not in old['body'], endpoint
    body = old['body'].rstrip() + '\n\n' + addition
    assert len(body) < 65536, endpoint
    fresh = api(endpoint)
    assert fresh['body'] == old['body'] and fresh['state'] == old['state'], endpoint
    if '/pulls/' in endpoint:
        assert fresh['draft'] == old['draft'] and fresh['head']['sha'] == old['head']['sha']
    payload = {'body': body}
    if title:
        payload['title'] = title
    api(endpoint, payload)
    result = api(endpoint)
    assert result['body'] == body and result['state'] == old['state'], endpoint
    if title:
        assert result['title'] == title
    if '/pulls/' in endpoint:
        assert result['draft'] == old['draft'] and result['head']['sha'] == old['head']['sha']
    receipts.append({'url': result['html_url'], 'state': result['state'], 'draft': result.get('draft'), 'head_sha': result.get('head', {}).get('sha'), 'before_body_sha256': sha(old['body']), 'after_body_sha256': sha(body), 'exact_body_readback': True, 'title': result['title']})
    (FAMILY / 'results' / ('published-' + endpoint.split('/')[-1] + '.md')).write_text(addition)
    (FAMILY / 'results/publication-receipt.json').write_text(json.dumps(receipts, indent=2) + '\n')
    print(result['html_url'], 'exact body/state readback passed', flush=True)

path = 'experiments/2026-10-23_0551_codex_adjudication-source-conservation/README.md'
remote = api('repos/pirl-unc/presto/contents/' + path + '?ref=82feaf2')
data = base64.b64decode(remote['content'])
assert data == (ROOT / path).read_bytes()
(FAMILY / 'results/report-readback.json').write_text(json.dumps({'url': REPORT, 'exact_bytes': True, 'git_blob_sha': remote['sha'], 'sha256': hashlib.sha256(data).hexdigest()}, indent=2) + '\n')
print('Published immutable report bytes match local', flush=True)
