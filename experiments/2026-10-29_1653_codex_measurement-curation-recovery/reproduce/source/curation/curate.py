"""Apply shared source curation to the frozen TRAIN cohort; CPU only."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
import modal

FID = '2026-10-29_1653_codex_measurement-curation-recovery'
PARENT = '2026-10-28_1540_codex_temporal-multi-output'
EXPECTED = 'c76aeafbb47c66b7fa904c191502de95f8812d45275cdfe680821838a37200b1'
app = modal.App('presto-measurement-curation-20261029')
volume = modal.Volume.from_name('presto-checkpoints')
image = modal.Image.debian_slim(python_version='3.12')
MODULES = ('modular.py', 'modular_quantitative.py', 'modular_ordinal.py', 'assay_reviews.py', 'assay_measurement_reviews.json', 'modular_hitlist.py', 'measurement_curation.py')
if modal.is_local():
    F = Path(__file__).resolve().parents[1]
    REPO = F.parents[1]
    for name in MODULES:
        image = image.add_local_file(REPO / 'data' / name, '/runtime/presto/data/' + name)
    image = image.add_local_file(Path('/Users/iskander/.cache/mhcseqs/mhc-full-seqs.csv'), '/bundle/catalog.csv')


def sha(path):
    with Path(path).open('rb') as h:
        return hashlib.file_digest(h, 'sha256').hexdigest()


@app.function(image=image, cpu=4, memory=16384, timeout=3600, max_containers=1, volumes={'/volume':volume})
def run():
    started = time.monotonic()
    # Namespace package only; this data-only worker does not import a model.
    sys.path.insert(0, '/runtime')
    from presto.data.modular_hitlist import HumanMHCCatalog, curated_purified_assays
    from presto.data.measurement_curation import POLICY, CensorScreenPolicy
    source = Path('/volume') / PARENT / 'revisions/kinetic-review-v2/prepared/hitlist-v4-iedb-train.jsonl'
    assert sha(source) == EXPECTED
    out = Path('/volume') / FID / 'curated'
    assert not out.exists(), 'Preserve earlier results'
    out.mkdir(parents=True)
    def records():
        with source.open() as stream:
            for line in stream:
                row = json.loads(line)
                assert row['split'] == 'train'
                yield row['provenance']['raw_record']
    with open('/bundle/catalog.csv') as stream:
        catalog = HumanMHCCatalog(csv.DictReader(stream))
    observations, dispositions, decisions, panels = curated_purified_assays(
        records, catalog, source_sha256=EXPECTED,
        dra_reference='2026-10-28_1540_codex_temporal-multi-output:declared-human-DR-reference',
    )
    counts = Counter()
    with source.open() as original, gzip.open(out / 'train.jsonl.gz', 'wt') as target:
        current_ordinal = -1
        for observation in observations:
            ordinal = observation.provenance['source_ordinal']
            while current_ordinal < ordinal:
                row = json.loads(next(original)); current_ordinal += 1
            assert row['provenance']['raw_record'] == observation.provenance['raw_record']
            assert row['endpoint'] == observation.endpoint and row['value'] == observation.value
            measurement = observation.provenance['measurement_curation']
            cls = row['mhc'][0].get('class_override') or row['mhc'][0]['mhc_class']
            counts[f'{cls}:{row["endpoint"]}:{row["qualifier"]}->{observation.qualifier}:{measurement["status"]}'] += 1
            row['qualifier'] = observation.qualifier
            row['provenance'].update({k:observation.provenance[k] for k in ('measurement_curation','assay_method_context','missing_qualifier_convention')})
            row['provenance']['measurement_curation_parent_sha256'] = EXPECTED
            target.write(json.dumps(row, allow_nan=False)+'\n')
    with gzip.open(out/'decisions.jsonl.gz','wt') as stream:
        for decision in decisions.values(): stream.write(json.dumps(asdict(decision),allow_nan=False)+'\n')
    (out/'panels.json').write_text(json.dumps(panels,indent=2,allow_nan=False)+'\n')
    summary = dict(policy=POLICY,screen=asdict(CensorScreenPolicy()),scope='frozen TRAIN physical cohort only; original validation/test unchanged',source_sha256=EXPECTED,catalog_sha256=sha('/bundle/catalog.csv'),dispositions=dict(dispositions),counts=dict(counts),decision_status_counts=dict(Counter(d.status for d in decisions.values())),panel_count=len(panels),seconds=time.monotonic()-started,cpu=4,memory_mib=16384,requested_gpu=None,files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir()})
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    archive=out.parent/'curated.tar.gz'
    with tarfile.open(archive,'w:gz') as tar: tar.add(out,arcname='curated')
    volume.commit()
    return dict(status='completed',archive=str(archive.relative_to('/volume')),sha256=sha(archive),bytes=archive.stat().st_size,summary=summary)


@app.local_entrypoint()
def main():
    f=Path(__file__).resolve().parents[1]
    assert not (f/'reproduce/curation-call.json').exists()
    snapshot=f/'reproduce/source/curation';snapshot.mkdir(parents=True)
    paths=[Path(__file__), *(f.parents[1]/'data'/name for name in MODULES)]
    for p in paths: shutil.copyfile(p,snapshot/p.name)
    launch=dict(argv=['python','-m','modal','run','--detach',__file__],git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),git_dirty=subprocess.check_output(['git','status','--short'],text=True),agent_model='Codex / GPT-6',source_hashes={p.name:sha(p) for p in paths},started_unix=time.time(),requested_gpu=None)
    (f/'reproduce/curation-launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    call=run.spawn();(f/'reproduce/curation-call.json').write_text(json.dumps(dict(call_id=call.object_id))+'\n')
    result=call.get();(f/'results/curation-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='summary'}),flush=True)

if __name__=='__main__':
    raise SystemExit('Use modal run --detach')
