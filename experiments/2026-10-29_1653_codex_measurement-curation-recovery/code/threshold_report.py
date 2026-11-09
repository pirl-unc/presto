"""Source/assay threshold table and conservative cross-study evidence summary."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import gzip
import json
import sys

F=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(F.parents[1]/'data'))
from measurement_curation import summarize_shared_limits
ROOT=Path('/Users/iskander/code/presto/artifacts')/F.name/'curated'


def write_csv(path, rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def main():
    panels=json.loads((F/'results/curation-panels.json').read_text())
    metadata={};scope={};wanted={tuple(p['panel']) for p in panels}
    with gzip.open(ROOT/'train.jsonl.gz','rt') as stream:
        for line in stream:
            r=json.loads(line);raw=r['provenance']['raw_record']
            key=(str(raw.get('reference_iri') or raw.get('pmid')),raw['mhc_class'],raw['mhc_restriction'],raw['assay_method'],raw['response_measured'])
            if key not in wanted:continue
            if key in scope:assert scope[key]==r['group_id']
            scope[key]=r['group_id'];metadata[key]=dict(reference=raw.get('reference_iri'),pmid=raw.get('pmid'),title=raw.get('reference_title'),submission=raw.get('submission_id'))
    assert set(scope)==wanted
    rows=[]
    for p in panels:
        key=tuple(p['panel']);m=metadata[key]
        rows.append(dict(direction=p['direction'],limit_nM=p['value_nM'],status=p['status'],effective_qualifier=p['effective_qualifier'] or 'withheld',affected_assay_rows=p['exact_heap_rows'],panel_assay_rows=p['panel_rows'],mhc_class=key[1],mhc=key[2],assay_method=key[3],quantity=key[4],reference=m['reference'],pmid=m['pmid'],source_component=scope[key],same_direction_explicit_bounds=p['same_direction_explicit_bounds'],opposing_bounds=p['opposite_direction_bounds'],qualitative_agreement=p['qualitative_agreement'],heap_fraction=p['heap_fraction'],neighbor_gap_ratio=p['neighbor_gap_ratio']))
    rows.sort(key=lambda r:(r['direction'],r['limit_nM'],r['reference'],r['mhc']))
    write_csv(F/'results/threshold-panels.csv',rows)
    shared=summarize_shared_limits(panels,scope)
    (F/'results/shared-threshold-evidence.json').write_text(json.dumps(shared,indent=2)+'\n')
    combined=defaultdict(list)
    for r in rows:combined[r['direction'],r['limit_nM'],r['quantity'],r['assay_method']].append(r)
    grouped=[]
    for (direction,value,quantity,method),items in sorted(combined.items()):
        grouped.append(dict(direction=direction,limit_nM=value,quantity=quantity,assay_method=method,inferred_assay_rows=sum(r['affected_assay_rows'] for r in items if r['status']=='inferred_panel_limit'),withheld_assay_rows=sum(r['affected_assay_rows'] for r in items if r['status']=='needs_review'),local_panels=len(items),independent_study_components=len({r['source_component'] for r in items}),classes=','.join(sorted({r['mhc_class'] for r in items})),references=sorted({r['reference'] for r in items}),pmids=sorted({str(r['pmid']) for r in items if r['pmid']})))
    write_csv(F/'results/threshold-summary.csv',grouped)
    lines=['# Source-scoped possible assay limits','',
           'Counts are distinct source assay rows, not donors or necessarily independent peptide–MHCs. All 71 panels are from the frozen TRAIN cohort. Inferred rows use inclusive inequalities; withheld rows have not been relabeled. The original explicit bounds and original values are retained.', '',
           'Each local rule is scoped to reference × MHC × reported method × literal quantity. Distinct nM values remain distinct, including nearby proxy thresholds. Shared-rule evidence counts linked study components, not allele panels; no cross-source rule is currently applied.', '',
           '## Limits and sources','',
           '| Limit nM | Direction | Quantity | Reported method | Classes | Inferred rows | Withheld rows | Panels / independent studies | Original IEDB references |',
           '|---:|---|---|---|---|---:|---:|---|---|']
    short={'half maximal inhibitory concentration (IC50)':'IC50','dissociation constant KD (~IC50)':'KD (~IC50)','dissociation constant KD (~EC50)':'KD (~EC50)','dissociation constant KD':'KD','half maximal effective concentration (EC50)':'EC50'}
    for r in grouped:
        refs=', '.join(f'[{x.rsplit("/",1)[-1]}]({x})' for x in r['references'])
        lines.append(f'| {r["limit_nM"]:.12g} | {r["direction"]} | {short.get(r["quantity"],r["quantity"])} | {r["assay_method"].replace("purified MHC/","")} | {r["classes"]} | {r["inferred_assay_rows"]} | {r["withheld_assay_rows"]} | {r["local_panels"]} / {r["independent_study_components"]} | {refs} |')
    lines += ['', '## Individual source/MHC panels','',
              '| Reference (PMID if supplied) | MHC | Quantity / reported method | Limit nM | Decision | Affected / panel rows | Same-direction explicit bounds | Qualitative agreement |',
              '|---|---|---|---:|---|---:|---:|---:|']
    for r in rows:
        ref=r['reference'];pmid=f' (PMID {r["pmid"]})' if r['pmid'] else ''
        decision=r['effective_qualifier'] if r['status']=='inferred_panel_limit' else 'review '+r['direction']
        lines.append(f'| [{ref.rsplit("/",1)[-1]}]({ref}){pmid} | {r["mhc"]} | {short.get(r["quantity"],r["quantity"])} / {r["assay_method"].replace("purified MHC/","")} | {r["limit_nM"]:.12g} | {decision} | {r["affected_assay_rows"]} / {r["panel_assay_rows"]} | {r["same_direction_explicit_bounds"]} | {r["qualitative_agreement"]:.1%} |')
    lines += ['', '## Stronger evidence before sharing','',
              'A shared review candidate requires ≥3 independent study components, explicit same-limit bounds in ≥2, ≥100 affected assay rows, ≥95% directional qualitative agreement, no opposing bounds, complete independence metadata, and a cited verified protocol-family mapping. A candidate still does not automatically change another source. Current method tags are unverified: **0 shared thresholds approved; 0 transfers applied**.', '',
              'The local screen requires ≥20 panel rows, ≥5 exact observations at an extreme and ≥10% of panel rows. Statistical inference requires same-direction explicit bounds at that extreme, or ≥20% heaping plus ≥80% matching qualitative outcomes; a floor additionally needs a ≥4-fold gap to the next value. Unsupported extreme heaps are withheld. These fixed screening thresholds are assumptions, not assay-protocol confirmation.', '',
              'This is an extreme-heap screen, not exhaustive recovery of all censoring: smaller panels, internally mixed assay ceilings and unsupported lower floors need source review. In particular, the 1 nM heaps remain withheld, and no universal 20,000/25,000/50,000 nM cap is imposed.', '']
    (F/'results/thresholds.md').write_text('\n'.join(lines))
    assert sum(r['inferred_assay_rows'] for r in grouped)==506
    assert sum(r['withheld_assay_rows'] for r in grouped)==618
    print(json.dumps(dict(local_panels=len(rows),summary_rows=len(grouped),shared_candidates=sum(r['status']=='shared_review_candidate' for r in shared),sources=len(set(scope.values())))))
if __name__=='__main__':main()
