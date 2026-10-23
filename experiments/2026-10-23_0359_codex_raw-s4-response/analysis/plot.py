from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((R/'results/outer-test-predictions.csv').open()))
metrics=json.loads((R/'results/pooled-metrics.json').read_text())
conditions=['mean','context','sequence'];labels=['Training-mean control','Nominal load + site','Load + site + sequence']
fig,axes=plt.subplots(1,3,figsize=(12,4.6),sharex=True,sharey=True)
for ax,condition,label in zip(axes,conditions,labels):
 selected=[r for r in rows if r['condition']==condition]
 x=[float(r['target']) for r in selected];y=[float(r['prediction']) for r in selected]
 ax.scatter(x,y,c=[int(r['outer_fold']) for r in selected],cmap='tab10',vmin=0,vmax=9,s=17,alpha=.65,linewidths=0)
 ax.plot([3,9],[3,9],color='#7b8794',linestyle='--',linewidth=1)
 ax.set(xlim=(3,9),ylim=(3,9),xlabel='Observed log10 peak area',title=label+'\nRMSE %.3f'%metrics[condition]['equal_peptide_rmse'])
 ax.set_aspect('equal');ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.14)
axes[0].set_ylabel('Held-out predicted log10 peak area')
fig.suptitle('Raw S4: nested peptide-grouped response diagnostic',fontsize=15)
fig.text(.5,.035,'224 peptide × site summaries from 22 selected peptides • colors identify outer folds\nRMSE weights each peptide equally; points are summaries, not independent source slots.',ha='center',fontsize=9,color='#52606d')
fig.subplots_adjust(left=.07,right=.98,top=.78,bottom=.20,wspace=.18)
fig.savefig(R/'results/held-out-response.png',dpi=180);fig.savefig(R/'results/held-out-response.pdf');plt.close(fig)
with (R/'results/summary.csv').open('w') as f:
 fields=['condition']+list(metrics['mean']);writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({'condition':c,**metrics[c]} for c in conditions)
