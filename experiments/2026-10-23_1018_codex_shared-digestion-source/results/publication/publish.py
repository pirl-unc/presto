import json
import subprocess
from pathlib import Path

ROOT=Path('/private/tmp/presto-modular-model/experiments/2026-10-23_1018_codex_shared-digestion-source/results/publication')
ROOT.mkdir(exist_ok=True)

def get(path):
    return json.loads(subprocess.check_output(['gh','api',path],text=True))

def append_verified(path, heading, addition, filename):
    before=get(path)
    assert before['state']=='open',path
    body=before['body']
    assert heading not in body, 'Refuse duplicate publication section'
    updated=body.rstrip()+'\n\n'+heading+'\n\n'+addition.strip()+'\n'
    assert get(path)['body']==body,'Concurrent edit; refresh before retry'
    payload=ROOT/(filename+'-request.json')
    payload.write_text(json.dumps({'body':updated}))
    subprocess.run(['gh','api',path,'--method','PATCH','--input',str(payload)],check=True,stdout=subprocess.DEVNULL)
    after=get(path)
    assert after['body']==updated and after['state']=='open'
    (ROOT/(filename+'-readback.json')).write_text(json.dumps(after,indent=2)+'\n')
    print(path,'exact body/open-state readback passed')

append_verified('repos/pirl-unc/hitlist/issues/770','## Shared-source candidate and kinetic qualification (2026-10-23)', '''
Shared source support is implemented in draft PR #773 (977adc2), based on merged #772. [Registered real-source report](https://github.com/pirl-unc/presto/blob/1503d67/experiments/2026-10-23_1018_codex_shared-digestion-source/README.md).

The actual 250-file package preserves all 218 prior source payloads byte-for-byte. All 11 new canonical relations and all 33 old baselines pass. Independent original-byte replay checks every selected workbook/FASTA/CSV record, independently decodes all 34 RData frames, validates typed numeric values/run joins/final slots and reproduces every canonical query hash. Original RF1 conflicts, forced zeros, missing values, modifications/splice claims and scoped enzyme/substrate/protocol metadata remain explicit. Format/lint, strict docs, 37 focused tests and the guarded normal suite (3,341 tests) pass. Final-head hosted CI/review/deployment remain pending.

There is still no registered training recipe. This replay validates source typing, not all underlying quantitative arithmetic or scientific correctness. Next: independently replay raw/background/technical aggregation across admitted substrates, exclude or reconcile RF1, define contiguous-product/mapping eligibility, evaluate repeated temporal signal and freeze source-group splits plus an MS-response/selection likelihood. No biological training has used the new source.

The separate [exact-versus-sampled kinetic experiment](https://github.com/pirl-unc/presto/blob/1503d67/experiments/2026-10-23_1021_codex_independent-cut-qualification/README.md) qualifies a minimal independent-cut process on synthetic data. It conserves residue mass and retains every product. Exact amounts match all cut-state sums to 1.11e-16; two-group score-gradient expectations agree to 1.82e-17. Both sampled group sizes recover known rates (median relative errors 3.43%/1.66% for 32/128); exact recovery is numerical precision. Independent NumPy replay checks all 10,000 updates, 30 noninitial checkpoints and 2,940 predictions. This fits rates directly with known response, not sequence-to-rate transfer or real APC concentrations. The full availability objective remains open.
''','issue770')

append_verified('repos/pirl-unc/presto/issues/131','## Shared digestion source and matched kinetic recovery', '''
[Source qualification](https://github.com/pirl-unc/presto/blob/1503d67/experiments/2026-10-23_1018_codex_shared-digestion-source/README.md) and [synthetic kinetic comparison](https://github.com/pirl-unc/presto/blob/1503d67/experiments/2026-10-23_1021_codex_independent-cut-qualification/README.md) are committed with complete frozen commands/source, checkpoints, predictions, raw probes, failed/corrected verification logs and canonical registry entries.

Hitlist draft [#773](https://github.com/pirl-unc/hitlist/pull/773) supplies original digestion evidence through the common source/query system; all prior payloads and every typed source row/value replay, and 3,341 local tests pass. Issue770 retains the pending quantitative-arithmetic, canonical recipe and biological observation gates.

The minimal kinetic process now has exact span expectations and sampled complete fragment partitions with identical support and residue conservation. On the prospectively registered noise-free five-rate fixture, exact recovery reaches numerical precision and sampled32/128 median rate errors are 3.43%/1.66%, versus 72.0% for one shared rate. Temporal test RMSE is approximately 0 / 0.00633 / 0.00277 versus 0.14176 copies per molecule. All 10,000 optimizer updates independently replay. A single-rate-control rank-tie reporting correction changes no fit or RMSE.

This qualifies mathematics and synthetic fitting only. Rates are learned directly; no sequence-to-rate encoder or biological digestion fit is claimed. The original real-data MAPP failures, IC50 issues, calibrated MS response, APC machinery/transport/clearance and cellular concentration goal remain unresolved. Keep this PR draft pending the full architecture and biological validation gates.
''','pr131')
