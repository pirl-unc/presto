"""Audit original S4 statistics and pairable normalization cells; no training."""
from pathlib import Path
import importlib.util,json,statistics,math
R=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('independent_xml',R/'reproduce/source/verify.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
book=next(iter(m.read_xml(Path('/private/tmp/hitlist-pxd004886-assets/supplementary-data-4.xlsx'))[0].values()))
get=lambda c:book.get(c,{}).get('value')
def col(n):
 out=''
 while n:n,d=divmod(n-1,26);out=chr(d+65)+out
 return out
raw={};norm={};mode=None
for i in range(1,75):
 label=get(f'A{i}')
 if isinstance(label,str) and label.strip().lower() in ['un-normalized','normalized']:mode=label.strip().lower()
 if isinstance(label,str) and '.' in label and mode:(raw if mode=='un-normalized' else norm)[label]=i
pairs=[];stat_checks=[]
for label,row in raw.items():
 for j in range(6,105):
  a=f'{col(j)}{row}';b=f'{col(j)}{norm[label]}';x=get(a);y=get(b)
  pairs.append({'source_label':label,'source_column':col(j),'raw_cell':a,'normalized_cell':b,'raw':x,'normalized':y,'ratio':y/x if m.numeric(x) and x>0 and m.numeric(y) and y>0 else None})
for mode,rows in [('raw',raw),('normalized',norm)]:
 for label,row in rows.items():
  values=[get(f'{col(j)}{row}') for j in range(6,105) if m.numeric(get(f'{col(j)}{row}'))]
  computed=[statistics.mean(values),statistics.stdev(values),100*statistics.stdev(values)/statistics.mean(values)] if len(values)>1 else [None]*3
  for c,value in zip('CDE',computed):
   source=get(f'{c}{row}')
   stat_checks.append({'mode':mode,'source_label':label,'source_cell':f'{c}{row}','reported':source,'recomputed':value,'relative_error':abs(value-source)/max(abs(source),1e-15) if m.numeric(value) and m.numeric(source) else None})
ratios=[]
for j in range(6,105):
 values=[p['ratio'] for p in pairs if p['source_column']==col(j) and p['ratio'] is not None]
 median=statistics.median(values) if values else None
 ratios.append({'source_column':col(j),'n_paired_reporters':len(values),'inferred_median_ratio':median,'max_relative_deviation_from_median':max(abs(x/median-1) for x in values) if median else None})
missing=[p for p in pairs if p['ratio'] is None and (m.numeric(p['raw']) or m.numeric(p['normalized']))]
summary={'matched_positive_pairs':sum(p['ratio'] is not None for p in pairs),'positive_normalized_without_positive_raw':sum(m.numeric(p['normalized']) and p['normalized']>0 and not(m.numeric(p['raw']) and p['raw']>0) for p in pairs),'positive_raw_without_positive_normalized':sum(m.numeric(p['raw']) and p['raw']>0 and not(m.numeric(p['normalized']) and p['normalized']>0) for p in pairs),'published_statistics_compared':sum(x['relative_error'] is not None for x in stat_checks),'max_statistics_relative_error':max(x['relative_error'] for x in stat_checks if x['relative_error'] is not None),'paired_columns':sum(x['n_paired_reporters']>0 for x in ratios),'columns_with_ratio_deviation_over1e_6':sum(x['max_relative_deviation_from_median'] is not None and x['max_relative_deviation_from_median']>1e-6 for x in ratios),'maximum_ratio_relative_deviation':max(x['max_relative_deviation_from_median'] for x in ratios if x['max_relative_deviation_from_median'] is not None),'scope':'Source-only arithmetic. Inferred ratios are diagnostics, not supplied normalization coefficients or source run identities. Missing cells are not zeros.'}
(R/'results/normalization-audit.json').write_text(json.dumps({'summary':summary,'unpaired_cells':missing,'column_diagnostics':ratios,'statistic_checks':stat_checks},indent=2)+'\n');print(json.dumps(summary),flush=True)
