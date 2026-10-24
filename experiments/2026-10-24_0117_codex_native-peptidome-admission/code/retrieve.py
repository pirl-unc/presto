"""Retrieve only registered public source assets; retain immutable byte receipts."""
import datetime
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
FAMILY=Path(__file__).resolve().parents[1]
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    c=json.loads((FAMILY/'reproduce/launch.json').read_text());r=json.loads((FAMILY/'reproduce/retrieval.json').read_text());destination=Path(c['asset_directory'])
    assert not (FAMILY/'results/retrieval.json').exists()
    def one(asset):
        tick=time.perf_counter();record=dict(asset,retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            response=requests.get(asset['url'],timeout=c['retrieval']['timeout_seconds_per_request'],headers={'User-Agent':'Presto-public-source-audit/1.0'},stream=True)
            record.update(status_code=response.status_code,final_url=response.url,content_type=response.headers.get('Content-Type'),content_disposition=response.headers.get('Content-Disposition'),etag=response.headers.get('ETag'))
            data=bytearray()
            for chunk in response.iter_content(1024*1024):
                data.extend(chunk)
                if len(data)>c['retrieval']['max_asset_bytes']:raise ValueError('registered asset size cap')
            path=destination/(asset['id']+'.bin');assert not path.exists();path.write_bytes(data)
            record.update(path=str(path),size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),first_bytes_hex=bytes(data[:16]).hex(),http_success=bool(response.ok),status='retrieved' if response.ok else 'http_error')
        except Exception as error:record.update(status='failed',error=repr(error))
        record['seconds']=time.perf_counter()-tick
        print(record['id'],record['status'],record.get('status_code'),record.get('size_bytes'),flush=True)
        return record
    # Independent read-only retrievals, four bounded workers. No source is executed.
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(one,r['assets']))
    dump(FAMILY/'results/retrieval.json',dict(assets=results,source_manifest=str(FAMILY/'reproduce/retrieval.json')))
if __name__=='__main__':main()
