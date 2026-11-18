from pathlib import Path
from copy import deepcopy
from collections import Counter,defaultdict
import json,random,hashlib,shutil
E=Path(__file__).resolve().parents[1];src=E/'results/cohort-v2';out=E/'results/cohort-v2-global';out.mkdir(exist_ok=True)
shutil.copytree(src/'data',out/'data',dirs_exist_ok=True)
for n in ('cohort.json','affinity-validation.jsonl'):shutil.copyfile(src/n,out/n)
rows=[json.loads(x) for x in (src/'data/train.jsonl').open()];pairs=defaultdict(list)
for i,r in enumerate(rows):pairs[r['provenance']['pair_id']].append(i)
buckets=defaultdict(list)
for ids in pairs.values():
 assert len(ids)==2 and sorted(rows[i]['value'] for i in ids)==[0.,1.]
 buckets[rows[ids[0]]['mhc'][0]['mhc_class']].append(ids)
fields=('cell_type','components','cytokines','interventions')
def payload(r):return {k:r['context'][k] for k in fields}
def fingerprint(r):return json.dumps(payload(r),sort_keys=True)
rng=random.Random(20261117);checks={}
for cls,members in buckets.items():
 before=Counter(fingerprint(rows[i[0]]) for i in members)
 contexts=[deepcopy(payload(rows[i[0]])) for i in members];rng.shuffle(contexts)
 for ids,context in zip(members,contexts,strict=True):
  for i in ids:
   rows[i]['context'].update(deepcopy(context))
   rows[i]['provenance']['context_control']='whole APC context shuffled across study/cell within MHC class'
 after=Counter(fingerprint(rows[i[0]]) for i in members)
 assert before==after
 assert all(fingerprint(rows[ids[0]])==fingerprint(rows[ids[1]]) for ids in members)
 checks[cls]={'pairs':len(members),'context_histogram_preserved':True,'pair_contexts_identical':True}
p=out/'data/train-shuffled-global.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
(out/'shuffle-verification.json').write_text(json.dumps(checks,indent=2))
receipt={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
(E/'reproduce/v2-global-corrected-inputs.json').write_text(json.dumps(receipt,indent=2));print(checks)
