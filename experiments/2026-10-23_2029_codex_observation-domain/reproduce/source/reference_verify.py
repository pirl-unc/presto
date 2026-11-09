"""Recompute every unperturbed full output with the independent NumPy solver."""
import json
import time
from pathlib import Path
import numpy as np
from reference import counts, features
from project import dump,sha

FAMILY=Path(__file__).resolve().parents[1]


def main():
    start=time.perf_counter()
    cfg=json.loads((FAMILY/'reproduce/reference-closure.json').read_text())
    for rel,expected in cfg['source_hashes'].items():assert sha(FAMILY/rel)==expected
    receipt=json.loads((FAMILY/'results/worker-receipts.json').read_text())
    assert len(receipt['workers'])==84
    cases=json.loads((FAMILY/'reproduce/cases.json').read_text())
    support=json.loads((FAMILY/'reproduce/training-support.json').read_text())
    coef=json.loads((FAMILY/'reproduce/coefficients.json').read_text())
    theta=np.array(support['initial_coefficients'])
    out=[]
    for case in cases:
        tick=time.perf_counter();cid=case['case_id']
        archive=FAMILY/f'results/independent-case-{cid:03d}.npz'
        assert not archive.exists(),'Refuse to overwrite independent reference'
        block=support['blocks'][case['block_id']]
        x=features(block['sequence'],block['context_log_enzyme'])
        ref=counts(x@theta,case['eta'],block['time_hours'],8,coef)
        raw=np.load(FAMILY/f'results/case-{cid:03d}.npz')
        actual=raw['counts_8']
        a,b=np.array(block['spans']).T;b=b-1
        target=ref[a,b]
        valid=bool(np.all((target>0)&np.isfinite(target)))
        log_error=float(np.max(np.abs(np.log(target)-raw['logs_8']))) if valid else None
        full_error=float(np.max(np.abs(ref-actual)))
        np.savez_compressed(archive,counts=ref)
        info=dict(**case,required_positive_finite=valid,required_log_max_abs=log_error,
            full_array_max_abs=full_error,full_reference_negative_count=int((ref<0).sum()),
            full_reference_zero_count=int((ref[np.triu_indices(len(ref))]==0).sum()),
            required_gate=valid and log_error<=cfg['required_log_tolerance'],
            full_array_diagnostic_pass=full_error<=cfg['full_array_absolute_diagnostic'],
            seconds=time.perf_counter()-tick,archive=str(archive.relative_to(FAMILY)),sha256=sha(archive))
        out.append(info)
        dump(FAMILY/'results/independent-reference.json',dict(status='running',cases=out))
    summary=dict(status='complete',cases=out,all_required_pass=all(r['required_gate'] for r in out),
        all_full_array_diagnostics_pass=all(r['full_array_diagnostic_pass'] for r in out),
        maximum_required_log_error=max(r['required_log_max_abs'] for r in out if r['required_log_max_abs'] is not None),
        maximum_full_array_error=max(r['full_array_max_abs'] for r in out),seconds=time.perf_counter()-start)
    dump(FAMILY/'results/independent-reference.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
