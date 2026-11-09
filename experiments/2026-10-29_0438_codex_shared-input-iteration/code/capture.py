"""Capture fixed TRAIN prefixes from an already verified, read-only staged cohort."""

# ruff: noqa: E501
# Preserve the executed remote capture program verbatim inside REMOTE.

from pathlib import Path
import asyncio
import hashlib
import json
import uuid
import importlib.metadata
import tarfile
import io
import time

F = Path(__file__).resolve().parents[1]
A = F.parents[1] / 'artifacts' / F.name
CONTAINER = 'ta-01M62XMD3MJH3A3KEE8GVPRNKR'
REMOTE = r'''
from pathlib import Path
import base64,hashlib,io,itertools,json,tarfile,time
root=Path('/runs/2026-10-28_1540_codex_temporal-multi-output/prepared')
manifest=json.loads((root/'wide-multi-s42.json').read_text())
names={'hitlist-v4-iedb-train.jsonl','hitlist_native_ms-train.jsonl','hitlist_ligand_background-train.jsonl'}
result=io.BytesIO(); sources=[]
with tarfile.open(fileobj=result,mode='w:gz') as archive:
 for source in manifest['sources']:
  path=Path(source['path'])
  if path.name not in names: continue
  with path.open('rb') as stream: payload=b''.join(itertools.islice(stream,5000))
  rows=[json.loads(line) for line in payload.splitlines()]
  assert len(rows)==5000 and all(row['split']=='train' for row in rows)
  info=tarfile.TarInfo(path.name);info.size=len(payload);archive.addfile(info,io.BytesIO(payload))
  sources.append(dict(**{k:v for k,v in source.items() if k not in ('path','sha256')},path=path.name,sha256=hashlib.sha256(payload).hexdigest(),full_source_path=str(path),declared_full_source_sha256=source['sha256'],rows=5000,bytes=len(payload)))
 assert len(sources)==3
 receipt=json.dumps(dict(sources=sources,captured_unix=time.time(),policy='first 5000 TRAIN lines per source; full-source hashes inherited from verified staging; no model evaluation'),indent=2).encode()
 info=tarfile.TarInfo('capture.json');info.size=len(receipt);archive.addfile(info,io.BytesIO(receipt))
payload=result.getvalue(); target=Path('/tmp/presto-shared-input-prefixes.tar.gz'); target.write_bytes(payload)
print(json.dumps(dict(path=str(target),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())))
'''


async def execute(command):
    # Modal 1.6.1 CLI waits for exit without draining its background STDOUT task.
    # Read explicit PIPE streams to EOF before accepting a transferred artifact.
    from modal.client import _Client
    from modal._utils.task_command_router_client import TaskCommandRouterClient
    from modal.container_process import _ContainerProcess
    from modal_proto import task_command_router_pb2 as proto

    client = await _Client.from_env()
    router = await TaskCommandRouterClient.init(client, CONTAINER)
    identifier = str(uuid.uuid4())
    await router.exec_start(
        proto.TaskExecStartRequest(
            task_id=CONTAINER,
            exec_id=identifier,
            command_args=command,
            stdout_config=proto.TaskExecStdoutConfig.TASK_EXEC_STDOUT_CONFIG_PIPE,
            stderr_config=proto.TaskExecStderrConfig.TASK_EXEC_STDERR_CONFIG_PIPE,
        )
    )
    process = _ContainerProcess(
        identifier, CONTAINER, client, command_router_client=router, text=False
    )
    stdout, stderr, code = await asyncio.gather(
        process.stdout.read(), process.stderr.read(), process.wait()
    )
    assert code == 0, stderr.decode(errors='replace')
    return stdout


async def main():
    assert importlib.metadata.version('modal') == '1.6.1'
    A.mkdir(parents=True, exist_ok=True)
    archive_path = A / 'prefixes.tar.gz'
    assert not archive_path.exists(), 'do not overwrite a capture'
    bundle = F / 'reproduce/source'
    (bundle / 'capture.remote.py').write_text(REMOTE)
    (bundle / 'capture.py').write_bytes(Path(__file__).read_bytes())
    remote = json.loads(await execute(['python', '-c', REMOTE]))
    code = (
        'from pathlib import Path; import sys; sys.stdout.buffer.write(Path('
        + repr(remote['path'])
        + ').read_bytes())'
    )
    payload = await execute(['python', '-c', code])
    assert len(payload) == remote['bytes']
    assert hashlib.sha256(payload).hexdigest() == remote['sha256']
    archive_path.write_bytes(payload)
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        for member in archive.getmembers():
            assert member.isfile() and Path(member.name).name == member.name
        archive.extractall(A, filter='data')
    receipt = json.loads((A / 'capture.json').read_text())
    for item in receipt['sources']:
        assert hashlib.sha256((A / item['path']).read_bytes()).hexdigest() == item['sha256']
    receipt.update(
        container=CONTAINER,
        local_artifacts=str(A),
        archive_sha256=hashlib.sha256(payload).hexdigest(),
        capture_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        remote_source_sha256=hashlib.sha256(REMOTE.encode()).hexdigest(),
        modal_version=importlib.metadata.version('modal'),
        transport='explicit SDK PIPE streams drained to EOF before size/hash validation',
        completed_unix=time.time(),
    )
    (F / 'results/capture.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(
        'Captured three TRAIN-only prefixes:',
        sum(s['rows'] for s in receipt['sources']),
        'rows,',
        sum(s['bytes'] for s in receipt['sources']),
        'bytes',
    )


if __name__ == '__main__':
    from modal._utils.async_utils import synchronizer

    synchronizer.create_blocking(main)()
