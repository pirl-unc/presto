from pathlib import Path
import json,gzip
R=Path(__file__).resolve().parents[1]
runs=json.loads(gzip.decompress((R/'results/checkpoints.json.gz').read_bytes()))
result=[]
for run in runs:
 if run['role']!='outer_test' or run['condition']=='mean':continue
 model=run['model'];names=[model['feature_names'][i] for i in model['kept_indices']]
 beta={name:b/scale for name,b,scale in zip(names,model['coefficients'],model['scale'])}
 result.append({'outer_fold':run['outer_fold'],'condition':run['condition'],'alpha':model['alpha'],'allocated_features':len(model['coefficients']),'dropped_features':model['dropped_features'],'raw_log_load_coefficient':beta['log10_nominal_load'],'train_rmse':run['train_metrics']['equal_peptide_rmse'],'outer_test_rmse':run['evaluation_metrics']['equal_peptide_rmse']})
(R/'results/coefficient-diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
