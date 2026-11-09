"""Fit mandatory temporal controls; write predictions without held-out outcomes."""

import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from model import ROOT,data,direct_controls,dump_json

assert json.loads((ROOT/'results/qualification.json').read_text())['passed']
datasets,parent,aas,index,sequences=data()
models,predictions,latest=direct_controls(datasets['train'],datasets)
out=ROOT/'results/controls';out.mkdir()
np.savez_compressed(out/'parameters.npz',latest=latest,sequences=np.array(sequences),
                    **{name+'_'+field:values[i] for name,values in models.items() for i,field in enumerate(('intercept','slope'))})
rows=[]
for name,prediction in predictions.items():
    for split,values in prediction.items():
        for i,(row,value) in enumerate(zip(datasets[split]['rows'],values,strict=True)):
            rows.append({'model':name,'split':split,'assignment_id':row['assignment_id'],'prediction':float(value),
                         'last_train_mean':float(latest[datasets[split]['sequence_index'][i]])})
pq.write_table(pa.Table.from_pylist(rows),out/'predictions.parquet')
dump_json(out/'receipt.json',{'prediction_sha256':hashlib.sha256((out/'predictions.parquet').read_bytes()).hexdigest(),
                             'n_models':len(models),'n_predictions':len(rows),'predictive_outcomes_read':False})
print('Saved control predictions',len(rows),flush=True)
