"""Validate recovered ligand forms through Hitlist and the common Presto adapter."""
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
import csv
import gzip
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
import modal

FID='2026-10-29_1653_codex_measurement-curation-recovery'
PARENT='2026-10-28_1540_codex_temporal-multi-output'
app=modal.App('presto-recovered-source-admission-20261029')
volume=modal.Volume.from_name('presto-checkpoints')
image=modal.Image.debian_slim(python_version='3.12')
MODULES=('modular.py','modular_quantitative.py','modular_ordinal.py','assay_reviews.py','assay_measurement_reviews.json','modular_hitlist.py','measurement_curation.py','modular_hitlist_ms.py')
if modal.is_local():
    F=Path(__file__).resolve().parents[1];REPO=F.parents[1]
    REC=Path('/Users/iskander/code/presto/artifacts')/FID/'recovered'
    WHEEL=Path('/Users/iskander/code/presto/.worktrees/hitlist-source-curation/dist/hitlist-2.1.1-py3-none-any.whl')
    image=image.add_local_file(WHEEL,'/bundle/'+WHEEL.name,copy=True).uv_pip_install('/bundle/'+WHEEL.name)
    for name in MODULES:image=image.add_local_file(REPO/'data'/name,'/runtime/presto/data/'+name)
    image=image.add_local_file(REC/'strazar_2023_recovered_unmodified.csv','/bundle/strazar_2023_recovered_unmodified.csv')
    image=image.add_local_file(REC/'source-evidence.jsonl','/bundle/source-evidence.jsonl')
    image=image.add_local_file(F/'results/strazar-recovery.json','/bundle/recovery.json')
    image=image.add_local_file(F.parent/PARENT/'results/study-selection.json','/bundle/study-selection.json')
    image=image.add_local_file(Path('/Users/iskander/.cache/mhcseqs/mhc-full-seqs.csv'),'/bundle/catalog.csv')


def sha(path):
    with Path(path).open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()


@app.function(image=image,cpu=4,memory=16384,timeout=3600,max_containers=1,volumes={'/volume':volume})
def run():
    started=time.monotonic();sys.path.insert(0,'/runtime')
    from hitlist.supplement import scan_supplementary
    from hitlist.provenance import ContributorCollector
    from presto.data.modular_hitlist import HumanMHCCatalog,AdmissionError
    from presto.data.modular_hitlist_ms import indexed_ms_observation
    import pandas as pd
    out=Path('/volume')/FID/'recovered-admission';assert not out.exists();out.mkdir(parents=True)
    recovery=json.loads(Path('/bundle/recovery.json').read_text())
    source=Path('/bundle/strazar_2023_recovered_unmodified.csv')
    assert sha(source)==recovery['output']['sha256']
    assert sha('/bundle/source-evidence.jsonl')==recovery['evidence']['sha256']
    entry=dict(file=source.name,pmid=37301199,source='Source-reviewed human monoallelic PSM intersection with published Stražar 2023 ligand set; see linked evidence ledger',defaults=dict(species='Homo sapiens'),sha256=sha(source),size_bytes=source.stat().st_size)
    with ContributorCollector(scratch_dir='/tmp') as collector:
        frame=scan_supplementary(entries=[entry],directory=source.parent,allow_download=False,provenance=collector).assign(source='supplement')
        contributor_metadata=collector.write([frame],out/'contributors.parquet')
    frame.to_parquet(out/'observations.parquet',index=False)
    with source.open() as stream:links={r['peptide']:{} for r in csv.DictReader(stream)}
    evidence={}
    with open('/bundle/source-evidence.jsonl') as stream:
        for line in stream:
            row=json.loads(line)
            if row['state']=='unmodified':evidence[row['peptide'],row['mhc_restriction']]=row
    with open('/bundle/catalog.csv') as stream:catalog=HumanMHCCatalog(csv.DictReader(stream))
    candidates=[];counts=Counter()
    # DataFrame JSON handles pandas' nullable scalar types before strict admission.
    for ordinal,raw in enumerate(json.loads(frame.to_json(orient='records'))):
        try:
            observation=indexed_ms_observation(raw,catalog,source_sha256=sha(source),ordinal=ordinal)
        except AdmissionError as error:
            counts[error.disposition+':'+error.reason]+=1;continue
        original=evidence[raw['peptide'],raw['mhc_restriction']]
        assert all(s['nterm']=='Hydrogen' and s['cterm']=='Free Acid' and not any(s[k] for k in ('modifications','variableSites','cys')) for s in original['support'])
        observation=replace(observation,provenance={**observation.provenance,'peptidoform_review':dict(evidence_id=original['evidence_id'],evidence_ledger_sha256=recovery['evidence']['sha256'],policy=recovery['policy'])})
        candidates.append(observation);counts['common_adapter_admitted']+=1
    selection=json.loads(Path('/bundle/study-selection.json').read_text())
    split_by_alias={a:c['split'] for c in selection['components'] for a in c['aliases']}
    assert split_by_alias.get('pmid:37301199','train')=='train'
    db=sqlite3.connect('file:/volume/'+PARENT+'/admission/admitted.sqlite?mode=ro',uri=True)
    wanted=sorted({r.peptide for r in candidates});embargo=set();prior_pmhc=set();same_study=set()
    for start in range(0,len(wanted),400):
        batch=wanted[start:start+400]
        for peptide,study,endpoint,payload in db.execute('SELECT peptide,study,endpoint,data FROM rows WHERE excluded=0 AND peptide IN ('+','.join('?'*len(batch))+')',batch):
            if split_by_alias.get(study) in ('validation','test'):embargo.add(peptide)
            if endpoint!='elution_observation':continue
            original=json.loads(payload)
            if original['value']!=1:continue
            for mhc in original['mhc']:
                key=(peptide,mhc['groove1'],mhc['groove2']);prior_pmhc.add(key)
                if study=='pmid:37301199':same_study.add(key)
    outputs=[]
    with gzip.open(out/'train.jsonl.gz','wt') as stream:
        for row in candidates:
            keys={(row.peptide,m.groove1,m.groove2) for m in row.mhc}
            if row.peptide in embargo:counts['heldout_peptide_embargo']+=1;continue
            if keys & same_study:counts['same_study_pmhc_duplicate']+=1;continue
            counts['pmhc_seen_in_other_studies' if keys & prior_pmhc else 'pmhc_not_seen_in_prior_admitted_positives']+=1
            row=replace(row,split='train',group_id='pmid:37301199',provenance={**row.provenance,'partition_status':'new older study TRAIN; existing held-out peptide embargo applied','partition_parent':PARENT,'prior_pmhc_overlap':bool(keys & prior_pmhc)})
            row.validate();stream.write(json.dumps(asdict(row),allow_nan=False)+'\n');outputs.append(row.observation_id)
    assert len(set(outputs))==len(outputs)
    result=dict(status='verified_candidate_cohort',counts=dict(counts),train_observations=len(outputs),source_sha256=sha(source),source_evidence_sha256=recovery['evidence']['sha256'],catalog_sha256=sha('/bundle/catalog.csv'),wheel_sha256=sha('/bundle/hitlist-2.1.1-py3-none-any.whl'),contributor_metadata=contributor_metadata,seconds=time.monotonic()-started,requested_gpu=None,files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir()},limitations='Only acquired and source-reviewed forms; no generated negatives and no retraining in this curation family. Existing fits remain unchanged.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    archive=out.parent/'recovered-admission.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname='recovered-admission')
    volume.commit();return dict(archive=str(archive.relative_to('/volume')),sha256=sha(archive),summary=result)


@app.local_entrypoint()
def main():
    f=Path(__file__).resolve().parents[1];callpath=f/'reproduce/recovery-admission-call.json';assert not callpath.exists()
    snapshot=f/'reproduce/source/recovery-admission';snapshot.mkdir(parents=True)
    for path in [Path(__file__),f/'code/recover_strazar.py',*(f.parents[1]/'data'/name for name in MODULES)]:shutil.copyfile(path,snapshot/path.name)
    launch=dict(argv=['python','-m','modal','run','--detach',__file__],git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),git_dirty=subprocess.check_output(['git','status','--short'],text=True),started_unix=time.time(),source_hashes={p.name:sha(p) for p in snapshot.iterdir()},wheel_sha256=sha(WHEEL),source_csv_sha256=sha(REC/'strazar_2023_recovered_unmodified.csv'),requested_gpu=None,cpu=4,memory_mib=16384)
    (f/'reproduce/recovery-admission-launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    call=run.spawn();callpath.write_text(json.dumps(dict(call_id=call.object_id))+'\n')
    result=call.get();(f/'results/recovery-admission.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary']['counts']))
