"""Static publication figures for the clean-calibrated benchmark."""
import json,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
data=json.load(open(ROOT/'results/clean_novelty_benchmark.json'))['records']
detectors=['knn_distance','lof_novelty','isolation_forest','one_class_svm']
labels=['kNN distance','LOF novelty','Isolation Forest','One-Class SVM']
colors=['#0072B2','#E69F00','#009E73','#CC79A7']; budgets=[.001,.005,.01,.02,.05]
fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax,rep in zip(axes,['control','stack']):
 for det,label,color in zip(detectors,labels,colors):
  means=[]; sds=[]; clean=[]
  rr=[r for r in data if r['representation']==rep]
  for b in budgets:
   means.append(np.mean([r['detectors'][det]['budgets'][str(b)]['poison_capture'] for r in rr]))
   sds.append(np.std([r['detectors'][det]['budgets'][str(b)]['poison_capture'] for r in rr]))
   clean.append(np.mean([r['detectors'][det]['budgets'][str(b)]['clean_removal_rate'] for r in rr]))
  ax.errorbar(np.array(clean)*100,np.array(means)*100,yerr=np.array(sds)*100,label=label,color=color,marker='o',capsize=2)
 ax.plot([0,6],[0,6],'--',color='0.65',lw=1); ax.set_title('60-feature control' if rep=='control' else '540-feature stack')
 ax.set_xlabel('Realized held-out clean removal (%)'); ax.grid(alpha=.2)
axes[0].set_ylabel('Poison capture (%)'); axes[1].legend(frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig(ROOT/'figures/clean_novelty_capture.png',dpi=300); plt.close(fig)
fig,axes=plt.subplots(1,4,figsize=(12,3),sharey=True)
families=['transpositions','block_reversal','block_swap','cyclic_shift']
for ax,fam in zip(axes,families):
 for x,rep,color in [(0,'control','#0072B2'),(1,'stack','#009E73')]:
  vals=[r['detectors']['knn_distance']['budgets']['0.05']['poison_capture']*100 for r in data if r['family']==fam and r['representation']==rep]
  ax.scatter(np.full(len(vals),x),vals,color=color,s=20); ax.errorbar(x,np.mean(vals),yerr=np.std(vals),fmt='s',color='black',capsize=3)
 ax.set_xticks([0,1],['60','540']); ax.set_title(fam.replace('_','\n')); ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('kNN poison capture at 5% budget (%)'); fig.tight_layout()
fig.savefig(ROOT/'figures/clean_novelty_family_variation.png',dpi=300); plt.close(fig)
print('wrote clean_novelty_capture.png and clean_novelty_family_variation.png')
