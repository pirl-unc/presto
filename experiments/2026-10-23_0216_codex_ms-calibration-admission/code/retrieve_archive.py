"""Retrieve the independent public Europe PMC supplementary archive."""
from pathlib import Path
import datetime,hashlib,json,urllib.request,zipfile,io,time,importlib.util
R=Path(__file__).resolve().parents[1];A=Path('/private/tmp/hitlist-pxd004886-assets')
url='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5566333/supplementaryFiles'
receipt={'url':url,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'maximum_bytes':64*1024*1024,'timeout_seconds':60}
start=time.monotonic()
try:
    with urllib.request.urlopen(url,timeout=60) as response:
        receipt.update(status=response.status,final_url=response.url,headers={k:v for k,v in response.headers.items() if k.lower() not in ('set-cookie','cookie','authorization')})
        parts=[];total=0
        while True:
            chunk=response.read(1024*1024)
            if not chunk:break
            total+=len(chunk)
            if total>receipt['maximum_bytes']:raise ValueError('Archive exceeds planned64MiB limit')
            parts.append(chunk)
    body=b''.join(parts);assert body[:2]==b'PK','Expected a ZIP archive'
    archive=A/'europepmc-supplements.zip';assert not archive.exists();archive.write_bytes(body)
    receipt.update(size_bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),archive_path=str(archive),members={})
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        assert z.testzip() is None
        receipt['archive_inventory']=[{'name':i.filename,'size_bytes':i.file_size} for i in z.infolist()]
        for number,object_id in {1:3,2:4,4:6,8:10}.items():
            wanted=f'41467_2017_249_MOESM{object_id}_ESM.xlsx'
            found=[x for x in z.namelist() if Path(x).name==wanted]
            assert len(found)==1,(wanted,found)
            data=z.read(found[0]);out=A/f'supplementary-data-{number}.xlsx'
            entry={'member':found[0],'sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data),'path':str(out)}
            if out.exists():
                assert out.read_bytes()==data,(wanted,'publisher/archive bytes differ')
                entry['existing_publisher_copy_identical']=True
            else:out.write_bytes(data)
            receipt['members'][str(number)]=entry
    receipt['success']=True
except Exception as exc:receipt.update(success=False,error=repr(exc))
receipt['wall_seconds']=time.monotonic()-start
(R/'results/archive-retrieval.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('headers','archive_inventory','members')}),flush=True)
if not receipt['success']:raise SystemExit(1)
spec=importlib.util.spec_from_file_location('source_inventory',R/'reproduce/source/launch.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ROOT=R
inventories={str(n):m.inventory(A/f'supplementary-data-{n}.xlsx') for n in [1,2,4,8]}
(R/'results/workbook-inventory-complete.json').write_text(json.dumps(inventories,indent=2,default=str)+'\n')
print('Full inventory',sum(x['n_source_cells'] for x in inventories.values()),'source cells',flush=True)
