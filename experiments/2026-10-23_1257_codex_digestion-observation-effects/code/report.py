"""Plot frozen training diagnostics; no fitting or selection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

F=Path(__file__).resolve().parents[1]
metrics=json.loads((F/"results/metrics.json").read_text())
shapes=json.loads((F/"results/shapes.json").read_text())
strata=json.loads((F/"results/baseline-strata.json").read_text())
layers=["corrected","restored","unthresholded"]
labels=["Corrected", "Restore baseline\nsame contributors", "Before censoring\nall contributors"]
fig,ax=plt.subplots(2,2,figsize=(12,8),layout="constrained")
x=np.arange(3);width=.35
for offset,predictor,name,color in [(-width/2,"no_change","No change","#9ca3af"),(width/2,"other_label_prediction","Other biological label","#287a99")]:
 values=[next(r["macro_rmse"] for r in metrics if r["parent"]=="all" and r["layer"]==l and r["predictor"]==predictor) for l in layers]
 bars=ax[0,0].bar(x+offset,values,width,label=name,color=color)
 ax[0,0].bar_label(bars,fmt="%.3f",fontsize=9,padding=3)
ax[0,0].set(xticks=x,xticklabels=labels,ylabel="Equal-parent log10-response RMSE",ylim=(0,1.16),title="A. Measured temporal repeatability")
ax[0,0].legend(frameon=False,fontsize=9)
for offset,kind,label,color in [(-width/2,"envelope","Growth envelope","#287a99"),(width/2,"concavity","Log-concavity","#ba6b45")]:
 vals=[100*shapes[l][kind]["0.3"]["equal_parent_fraction"] for l in layers]
 bars=ax[0,1].bar(x+offset,vals,width,label=label,color=color)
 ax[0,1].bar_label(bars,fmt="%.2f",fontsize=9,padding=3)
ax[0,1].set(xticks=x,xticklabels=labels,ylabel="Equal-parent violation fraction (%)",ylim=(0,10.5),title="B. Shape sensitivity (>0.3 log10)")
ax[0,1].legend(frameon=False,fontsize=9)
x=np.arange(len(strata));bins=[f"{s['lower']:.2g}–{s['upper']:.2g}" for s in strata]
bars=ax[1,0].bar(x,[100*s["weight_mass"] for s in strata],color="#8f9b65")
ax[1,0].set(xticks=x,xticklabels=bins,yscale="log",ylabel="Share of equal-parent weight (%)",title="C. Reference baseline fraction")
ax[1,0].set_xlabel("Subtracted baseline / retained pre-subtraction signal")
for key,label,color in [("saved_model_rmse","Previously fitted kinetic model","#ba6b45"),("repeatability_rmse","Other biological label","#287a99")]:
 ax[1,1].plot(x,[s[key] for s in strata],"o-",label=label,color=color)
ax[1,1].set(xticks=x,xticklabels=bins,ylabel="Corrected-response RMSE",title="D. Error by reference baseline fraction")
ax[1,1].set_xlabel("Last bin contains only 3 response observations")
ax[1,1].legend(frameon=False,fontsize=9)
fig.suptitle("Training-only observation diagnostics — 6 substrates, unchanged admitted support",fontsize=14)
fig.savefig(F/"results/observation-effects.png",dpi=170)
fig.savefig(F/"results/observation-effects.pdf")
print("Two plots written; no optimizer or held-out evaluation")
