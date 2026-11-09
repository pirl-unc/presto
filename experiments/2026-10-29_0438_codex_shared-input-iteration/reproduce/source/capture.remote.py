
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
