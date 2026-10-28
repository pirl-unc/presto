"""Source-semantics smoke check only; not fresh membership or predictive evaluation."""
from pathlib import Path
from collections import Counter
import csv,json,hashlib,time
import pyarrow.parquet as pq
from presto.data.modular_hitlist import HumanMHCCatalog,AdmissionError
from presto.data.modular_hitlist_ms import indexed_ms_observation

F=Path(__file__).resolve().parents[1]
source=Path('/Users/iskander/.hitlist/corrected-corpus/public-downloads/v4/bundles/2026-10-16T07-20-43Z/other_assays.parquet')
digest=hashlib.sha256(source.read_bytes()).hexdigest()
with open('/Users/iskander/.cache/mhcseqs/mhc-full-seqs.csv') as stream:catalog=HumanMHCCatalog(csv.DictReader(stream))
counts=Counter();studies=Counter();n=0
for batch in pq.ParquetFile(source).iter_batches(batch_size=8192):
    for raw in batch.to_pylist():
        if raw.get('assay_modality')=='ms':
            try:
                row=indexed_ms_observation(raw,catalog,source_sha256=digest,ordinal=n,dra_reference=F.name+':human_dr_dra0101_v1')
                assert row.value==0 and row.provenance['label_kind']=='reported_nondetection'
                counts['admitted:'+row.mhc[0].mhc_class]+=1
                studies[str(raw.get('pmid'))]+=1
            except AdmissionError as exc:counts[exc.disposition+':'+exc.reason]+=1
        n+=1
result=dict(status='passed',scope='cached-index source-semantics compatibility only; no membership or prediction reused',source=str(source),sha256=digest,scanned=n,counts=dict(counts),study_counts=dict(studies),completed_unix=time.time())
(F/'results/cached-negative-admission-smoke.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'scanned':n,'counts':counts},indent=2))
