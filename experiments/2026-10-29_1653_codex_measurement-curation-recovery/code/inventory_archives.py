"""Count released measurements without promoting transformed or generated labels."""
from collections import Counter
from pathlib import Path
import bz2
import csv
import io
import json
import tarfile
import time
import zipfile

F=Path(__file__).resolve().parents[1]

def csv_inventory(stream):
    reader=csv.DictReader(stream)
    fields=reader.fieldnames;counts={k:Counter() for k in fields if k in ('measurement_source','measurement_type','measurement_kind','measurement_inequality','source','original_allele')}
    count=0
    for row in reader:
        count+=1
        for k,c in counts.items():c[row[k]]+=1
    return dict(rows=count,fields=fields,strata={k:dict(c) for k,c in counts.items()})

def main():
    started=time.monotonic();out=[]
    receipts=[F/'results/acquisition.json',F/'results/acquisition-dtu.json',F/'results/acquisition-atlas.json']
    sources=[r for p in receipts if p.exists() for r in json.loads(p.read_text()) if r['status']=='downloaded']
    for spec in sources:
        path=Path(spec['path']);entry=dict(id=spec['id'],archive_sha256=spec['sha256'],members=[])
        if path.name.endswith(('.tar.bz2','.tar.gz')):
            with tarfile.open(path) as tar:
                for member in tar:
                    if not member.isfile():continue
                    name=member.name;item=dict(name=name,bytes=member.size)
                    if 'mhcflurry-data_curated'==spec['id'] and name.endswith('curated_training_data.affinity.csv.bz2'):
                        item.update(csv_inventory(io.TextIOWrapper(bz2.BZ2File(tar.extractfile(member)))))
                    if spec['id'].startswith('netmhc') and name.rsplit('_',1)[-1] in ('ba','el'):
                        counts=Counter();total=0
                        for line in tar.extractfile(member):
                            tokens=line.split()
                            if len(tokens)<3:continue
                            total+=1;value=float(tokens[1]);counts['zero' if value==0 else 'one' if value==1 else 'interior']+=1
                        item.update(rows=total,target_bins=dict(counts),interpretation='BA transformed concentration; EL positive/generated background. Not source-native assay identity.')
                    entry['members'].append(item)
        elif path.suffix=='.zip':
            with zipfile.ZipFile(path) as archive:entry['members']=[dict(name=m.filename,bytes=m.file_size) for m in archive.infolist() if not m.is_dir()]
        elif spec['id']=='netmhcstabpan-stability':
            values=Counter();rows=0;alleles=set()
            with path.open() as stream:
                next(stream)
                for line in stream:
                    a,p,v=line.split();rows+=1;alleles.add(a);values[float(v)]+=1
            entry.update(rows=rows,alleles=len(alleles),unit='hours',zero_rows=values[0.0],common_values=values.most_common(12))
        else:continue
        out.append(entry);(F/'results/archive-inventory.json').write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps(dict(id=entry['id'],members=len(entry['members']),rows=entry.get('rows'))),flush=True)
    print('seconds',time.monotonic()-started)
if __name__=='__main__':main()
