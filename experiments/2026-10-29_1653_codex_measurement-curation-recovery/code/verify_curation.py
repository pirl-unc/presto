"""Verify collected decisions and effective qualifiers against preserved raw rows."""
from collections import Counter
from pathlib import Path
import gzip
import hashlib
import json
import sys
import tarfile

F=Path(__file__).resolve().parents[1]
ROOT=Path('/Users/iskander/code/presto/artifacts')/F.name
sys.path.insert(0,str(F.parents[1]/'data'))
from measurement_curation import CensorDecision,curated_measurement

def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    receipt=json.loads((F/'results/curation-receipt.json').read_text());archive=ROOT/'curated.tar.gz'
    assert sha(archive)==receipt['sha256']
    with tarfile.open(archive) as tar: tar.extractall(ROOT,filter='data')
    root=ROOT/'curated';summary=json.loads((root/'summary.json').read_text())
    for name,meta in summary['files'].items(): assert sha(root/name)==meta['sha256']
    with gzip.open(root/'decisions.jsonl.gz','rt') as stream:
        decisions={r.assay_iri:r for line in stream for r in [CensorDecision(**json.loads(line))]}
    counts=Counter();seen=set();changes=Counter();context=Counter()
    with gzip.open(root/'train.jsonl.gz','rt') as stream:
        for line in stream:
            r=json.loads(line);assert r['split']=='train';raw=r['provenance']['raw_record'];identifier=raw['assay_iri']
            assert identifier not in seen;seen.add(identifier)
            if r['endpoint']=='qualitative_binding':
                assert r['qualifier']=='=';continue
            measurement=curated_measurement(raw,decisions.get(identifier))
            assert r['qualifier']==measurement['effective_qualifier']
            assert r['value']==raw['quantitative_value']
            assert r['provenance']['measurement_curation']==measurement
            counts[r['endpoint']+':'+measurement['status']]+=1
            context[r['provenance']['assay_method_context']['review_status']]+=1
            if identifier in decisions:
                d=decisions[identifier];assert d.status!='needs_review'
                changes[d.evidence['direction']]+=1
    assert len(seen)==sum(v for k,v in summary['dispositions'].items() if k.startswith('admitted:'))
    assert all((key not in seen)==(d.status=='needs_review') for key,d in decisions.items())
    result=dict(status='passed',admitted_rows=len(seen),withheld_review_rows=sum(d.status=='needs_review' for d in decisions.values()),inferred_bounds=dict(changes),by_endpoint_status=dict(counts),method_context=dict(context),archive_sha256=sha(archive),source_values_and_raw_records_preserved=True,validation_test_untouched=True)
    (F/'results/curation-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    for name in ('summary.json','panels.json'):
        (F/'results'/('curation-'+name)).write_bytes((root/name).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k not in ('by_endpoint_status',)}))
if __name__=='__main__':main()
