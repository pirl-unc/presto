"""Collect closed compact artifacts; large raw data stays on the named volume."""
from pathlib import Path
import argparse,json,hashlib
import modal
from common import NAME,E


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation');p.add_argument('--fits');p.add_argument('--files');p.add_argument('--controls');args=p.parse_args()
    volume=modal.Volume.from_name('presto-checkpoints');receipts=[]
    def get(relative):
        target=E/'results'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix(target.suffix+'.download')
        h=hashlib.sha256()
        try:
            with tmp.open('wb') as out:
                for block in volume.read_file(NAME+'/'+relative):out.write(block);h.update(block)
            tmp.replace(target)
        except BaseException:
            tmp.unlink(missing_ok=True);raise
        receipts.append({'remote':NAME+'/'+relative,'local':str(target),'sha256':h.hexdigest(),'bytes':target.stat().st_size})
        return target
    for run in (args.evaluation or '').split(','):
        if not run:continue
        prefix='evaluation/'+run
        report=json.loads(get(prefix+'/receipt.json').read_text())
        for population in report['metrics']:
            get(prefix+'/'+population+'-metrics.json');get(prefix+'/'+population+'-predictions.jsonl.gz')
    for run in (args.fits or '').split(','):
        if not run:continue
        prefix='fits/'+run;report=json.loads(get(prefix+'/receipt.json').read_text())
        for name in ['hardware.json','manifest.json','source-staging.json','actual-training-exposure.json','update-check.json','raw/summary.json','raw/history.json','raw/resolved-config.json','raw/coverage.json','raw/selected.pt']:
            path=get(prefix+'/'+name)
            if name=='raw/selected.pt':assert hashlib.sha256(path.read_bytes()).hexdigest()==report['checkpoint_sha256']
        if json.loads((E/'results'/prefix/'manifest.json').read_text())['training'].get('processing_pair_training'):
            get(prefix+'/raw/processing-pair-draws.json')
    for run in (args.controls or '').split(','):
        if not run:continue
        prefix='controls/'+run
        report=json.loads(get(prefix+'/receipt.json').read_text())
        for filename in report['files']:get(prefix+'/'+filename)
    for name in (args.files or '').split(','):
        if name:get(name)
    path=E/'reproduce/collected.jsonl'
    with path.open('a') as f:
        for receipt in receipts:f.write(json.dumps(receipt)+'\n')
    print(json.dumps({'downloaded':len(receipts),'bytes':sum(r['bytes'] for r in receipts)}))

if __name__=='__main__':main()
