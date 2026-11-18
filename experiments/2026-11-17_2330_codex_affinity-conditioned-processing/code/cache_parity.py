"""Numerical compatibility with pre-existing pinned external predictor caches."""
from pathlib import Path
import hashlib,json,os,subprocess,tempfile,time
import modal
from common import NAME,PRIOR,E,VOLUME,sha,dump
from data_pool import external_image,ensure_external
app=modal.App('presto-external-affinity-cache-parity')

@app.function(image=external_image,cpu=2,memory=12288,timeout=7200,volumes={'/volume':VOLUME})
def verify():
    ensure_external();start=time.monotonic();caches=[]
    for path in (Path('/volume')/PRIOR/'external').glob('*.json'):
        r=json.loads(path.read_text());caches.append((r,path))
    records=[]
    with tempfile.TemporaryDirectory() as td:
        for cls in ('I','II'):
            candidates=sorted([x for x in caches if x[0]['class']==cls],key=lambda x:(-len(x[0]['scores']),x[0]['allele']))[:4]
            for r,path in candidates:
                peptides=sorted(r['scores'],key=lambda p:hashlib.sha256(p.encode()).hexdigest())[:100]
                input_path=Path(td)/'peptides';input_path.write_text('\n'.join(peptides)+'\n')
                package,wrapper,version=('netMHCpan-4.2','netMHCpan','4.2c') if cls=='I' else ('netMHCIIpan-4.3','netMHCIIpan','4.3e')
                cmd=[f'/net/{package}/{wrapper}','-p' if cls=='I' else '-f',str(input_path),'-a',r['allele'],'-BA']+([] if cls=='I' else ['-inptype','1'])
                result=subprocess.run(cmd,env=dict(os.environ,NETMHC_BUNDLE_TMPDIR=td),capture_output=True,text=True,timeout=3600)
                assert result.returncode==0 and ('version '+version) in result.stdout
                header=None;parsed={}
                for line in result.stdout.splitlines():
                    cells=line.split()
                    if cells and cells[0]=='Pos' and 'Peptide' in cells:header=cells
                    elif header and cells and cells[0].isdigit():
                        d=dict(zip(header,cells))
                        parsed[d['Peptide']]={'ba_nm':float(next(d[k] for k in ('Aff(nM)','Affinity(nM)','Affinity') if k in d)),
                            'ba_rank':float(d['%Rank_BA']),'el_score':float(d['Score_EL']),'el_rank':float(d['%Rank_EL'])}
                assert set(parsed)==set(peptides)
                deltas={key:max(abs(parsed[p][key]-r['scores'][p][key]) for p in peptides) for key in ('ba_nm','ba_rank','el_score','el_rank')}
                records.append({'class':cls,'allele':r['allele'],'rows':len(peptides),'version':version,'cache_sha256':sha(path),'cache':str(path),'max_delta':deltas})
                assert all(v==0 for v in deltas.values()),records[-1]
    report={'status':'passed','records':records,'rows':sum(r['rows'] for r in records),'seconds':time.monotonic()-start,'test_accessed':False,
            'scope':'Deterministic numerical spot check, four largest immutable old-cache alleles/class,100 queries/allele; does not certify comparator training independence'}
    dump(Path('/volume')/NAME/'predictor-cache-parity.json',report);VOLUME.commit();return report

@app.local_entrypoint()
def main():
    call=verify.spawn();dump(E/'reproduce/cache-parity-call.json',{'call':call.object_id})
    result=call.get();dump(E/'results/predictor-cache-parity.json',result);print(json.dumps(result),flush=True)
