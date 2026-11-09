"""Pin original calibration sources and audit workbook structure, without curation."""
from __future__ import annotations
import concurrent.futures
import datetime
import gzip
import hashlib
import json
import resource
import time
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import openpyxl

ROOT=Path(__file__).resolve().parents[1]
ASSETS=Path('/private/tmp/hitlist-pxd004886-assets')
BASE='https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-017-00249-5/MediaObjects/'
SOURCES={1:3,2:4,4:6,8:10}
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
def sha(b):return hashlib.sha256(b).hexdigest()
def retrieve(pair):
    data_id,object_id=pair
    url=BASE+f'41467_2017_249_MOESM{object_id}_ESM.xlsx'
    dest=ASSETS/f'supplementary-data-{data_id}.xlsx'
    receipt={'supplementary_data':data_id,'url':url,'path':str(dest),'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        if dest.exists():raise FileExistsError(f'Original already exists: {dest}; do not overwrite')
        with urllib.request.urlopen(url,timeout=60) as response:
            body=response.read()
            receipt.update(status=response.status,final_url=response.url,headers=dict(response.headers))
        assert body[:2]==b'PK','Expected XLSX ZIP, not an error page'
        with zipfile.ZipFile(__import__('io').BytesIO(body)) as z:assert z.testzip() is None
        dest.write_bytes(body)
        receipt.update(size_bytes=len(body),sha256=sha(body),success=True)
    except Exception as exc:receipt.update(success=False,error=repr(exc))
    return receipt

def inventory(path):
    cached=openpyxl.load_workbook(path,data_only=True,read_only=False)
    formula=openpyxl.load_workbook(path,data_only=False,read_only=False)
    result={'sheets':[],'openpyxl_version':openpyxl.__version__}
    cells=[]
    with zipfile.ZipFile(path) as z:
        wb=ET.fromstring(z.read('xl/workbook.xml'))
        rels={r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        for sheet in wb.findall('s:sheets/s:sheet',NS):
            name=sheet.attrib['name'];ws=cached[name];wf=formula[name]
            target=rels[sheet.attrib['{'+NS['r']+'}id']]
            target=target.lstrip('/') if target.startswith('/') else 'xl/'+target
            xml=ET.fromstring(z.read(target))
            xml_populated={c.attrib['r']:c for c in xml.findall('.//s:sheetData/s:row/s:c',NS) if c.find('s:v',NS) is not None or c.find('s:is',NS) is not None or c.find('s:f',NS) is not None}
            py_populated={c.coordinate:c for row in wf for c in row if c.value is not None}
            # Empty formula-cache cells and explicit empty strings remain source records.
            missing=set(py_populated)-set(xml_populated)
            assert not missing,(name,missing)
            xform={k:c.find('s:f',NS).text for k,c in xml_populated.items() if c.find('s:f',NS) is not None}
            pform={k:c.value for k,c in py_populated.items() if c.data_type=='f'}
            assert set(pform)==set(xform),(name,'formula coordinates differ')
            # Shared/array formula expansion is explicitly recorded, not silently assumed equal.
            fdelta=[k for k in pform if xform[k] is not None and pform[k]!='='+xform[k]]
            assert not fdelta,(name,'formula text differs',fdelta[:3])
            for coord,c in xml_populated.items():
                pc=wf[coord];cv=ws[coord]
                cells.append({'sheet':name,'coordinate':coord,'value':cv.value,'formula_or_value':pc.value,'data_type':pc.data_type,'number_format':pc.number_format,'xml':ET.tostring(c,encoding='unicode')})
            preview=[[{'cell':c.coordinate,'value':c.value} for c in row if c.value is not None] for row in ws.iter_rows(min_row=1,max_row=min(ws.max_row,22))]
            result['sheets'].append({'name':name,'state':ws.sheet_state,'max_row':ws.max_row,'max_column':ws.max_column,'xml_populated_cells':len(xml_populated),'openpyxl_populated_cells':len(py_populated),'formula_cells':len(xform),'formula_cache_missing':sum(ws[k].value is None for k in xform),'xml_only_coordinates':sorted(set(xml_populated)-set(py_populated)),'merged_ranges':[str(x) for x in ws.merged_cells.ranges],'preview':preview})
    dest=ROOT/'results'/f'{path.stem}-cells.jsonl.gz'
    with gzip.GzipFile(filename='',mode='wb',fileobj=dest.open('wb'),mtime=0) as f:
        for record in cells:f.write((json.dumps(record,default=str)+'\n').encode())
    result.update(cell_dump=str(dest.relative_to(ROOT)),cell_dump_sha256=sha(dest.read_bytes()),n_source_cells=len(cells),independent_structure_check='passed')
    return result

def main():
    started=time.monotonic();ASSETS.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:receipts=list(pool.map(retrieve,SOURCES.items()))
    (ROOT/'results/retrieval.json').write_text(json.dumps(receipts,indent=2)+'\n')
    inventories={}
    for receipt in receipts:
        if receipt['success']:inventories[str(receipt['supplementary_data'])]=inventory(Path(receipt['path']))
    (ROOT/'results/workbook-inventory.json').write_text(json.dumps(inventories,indent=2,default=str)+'\n')
    summary={'retrieval_complete':all(r['success'] for r in receipts),'files':len(inventories),'wall_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'python':__import__('sys').version,'n_sheets':sum(len(x['sheets']) for x in inventories.values()),'n_source_cells':sum(x['n_source_cells'] for x in inventories.values()),'training':'none; source structure audit only'}
    (ROOT/'results/execution.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    if not summary['retrieval_complete']:raise SystemExit(1)
if __name__=='__main__':main()
