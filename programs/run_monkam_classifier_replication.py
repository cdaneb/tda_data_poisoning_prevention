"""Duplicate-safe replication of the supplied UNSW Random Forest results."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from programs.data_loader import load_unsw
from programs.monkam_representation import equivalence_profile,stable_hash
from programs.monkam_workbook import load_numeric_workbook
from programs.resource_usage import peak_rss_kib

RESULTS=ROOT/'results'; PRE=RESULTS/'monkam_classifier_replication_preregistration.json'
OUT=RESULTS/'monkam_classifier_replication.json'; CSVOUT=RESULTS/'monkam_classifier_replication_summary.csv'
SPLITS=RESULTS/'monkam_classifier_splits'; SEEDS=[60,42,123,456,789]; WORKERS=8

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def lock():
 d={"experiment":"monkam_duplicate_safe_classifier_replication","inputs":{
  "raw":sha(ROOT/'data/Payload_data_UNSW.csv'),"tda280":sha(ROOT/'monkam_files/tda_280_x_y.xlsx')},
  "eligible_indices":list(range(79881)),"eligible_indices_hash":stable_hash(list(range(79881))),
  "seeds":SEEDS,"parent_poison_relationships":"no delivered poison rows are in this classifier population",
  "tasks":{"multiclass":{"labels":"10 original labels","test_size":.2},
           "binary":{"labels":"normal versus malicious","test_size":.1}},
  "representations":["raw 1500 bytes","supplied 280 TDA features"],
  "model":{"class":"RandomForestClassifier","notebook":"all sklearn defaults, unseeded",
           "controlled":{"n_estimators":100,"random_state":"seed","n_jobs":WORKERS}},
  "protocols":["random_row","group_raw_payload","group_tda_vector"],"preprocessing":"none",
  "primary_endpoints":["accuracy","macro_f1","weighted_f1","balanced_accuracy"],
  "secondary_endpoints":["per-class precision/recall","confusion matrix","overlap","novel/repeated accuracy"],
  "permitted_sensitivity":["none; conflicting labels retained"],
  "gates":{"row_alignment":True,"zero_group_overlap_for_group_protocols":True},"workers":WORKERS}
 d['lock_hash']=stable_hash(d); return d
def row_keys(X): return np.asarray([hashlib.sha256(np.ascontiguousarray(r).view(np.uint8)).hexdigest() for r in X])
def make_split(n,groups,test_size,seed,protocol):
 idx=np.arange(n)
 if protocol=='random_row': return train_test_split(idx,test_size=test_size,random_state=seed)
 return next(GroupShuffleSplit(1,test_size=test_size,random_state=seed).split(idx,groups=groups))
def score(ytrue,ypred,classes):
 p,r,_,support=precision_recall_fscore_support(ytrue,ypred,labels=classes,zero_division=0)
 return {"accuracy":accuracy_score(ytrue,ypred),"macro_f1":f1_score(ytrue,ypred,average='macro'),
  "weighted_f1":f1_score(ytrue,ypred,average='weighted'),"balanced_accuracy":balanced_accuracy_score(ytrue,ypred),
  "per_class":{str(c):{"precision":p[i],"recall":r[i],"support":int(support[i])} for i,c in enumerate(classes)},
  "confusion_matrix":confusion_matrix(ytrue,ypred,labels=classes).tolist()}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--prepare-only',action='store_true'); ap.add_argument('--seed',default='all'); a=ap.parse_args()
 design=lock(); RESULTS.mkdir(exist_ok=True); SPLITS.mkdir(exist_ok=True)
 if a.prepare_only: PRE.write_text(json.dumps(design,indent=2,sort_keys=True)+'\n'); print(PRE); return
 if not PRE.exists() or stable_hash(json.load(open(PRE)))!=stable_hash(design): raise RuntimeError('preregistration mismatch')
 seeds=SEEDS if a.seed=='all' else [int(a.seed)]; started=time.time()
 Xraw,y=load_unsw(); Xtda,ybook,_=load_numeric_workbook(ROOT/'monkam_files/tda_280_x_y.xlsx')
 if len(Xraw)!=len(Xtda) or not np.array_equal(y,ybook): raise AssertionError('workbook/raw row alignment failed')
 rawkeys=row_keys(Xraw); tdakeys=row_keys(Xtda); tasks={'multiclass':y,'binary':np.where(y=='normal','normal','malicious')}
 records=[]
 for seed in seeds:
  for task,ytask in tasks.items():
   test_size=.2 if task=='multiclass' else .1; classes=np.unique(ytask)
   for protocol,groups in [('random_row',None),('group_raw_payload',rawkeys),('group_tda_vector',tdakeys)]:
    tr,te=make_split(len(y),groups,test_size,seed,protocol)
    split_hash=stable_hash({"train":tr.tolist(),"test":te.tolist()})
    overlap_raw=len(set(rawkeys[tr])&set(rawkeys[te])); overlap_tda=len(set(tdakeys[tr])&set(tdakeys[te]))
    for rep,X,keys in [('raw',Xraw,rawkeys),('tda280',Xtda,tdakeys)]:
     cell=SPLITS/f'{task}_{protocol}_{rep}_seed{seed}.json'
     if cell.exists(): records.append(json.load(open(cell))); continue
     t=time.time(); model=RandomForestClassifier(random_state=seed,n_jobs=WORKERS).fit(X[tr],ytask[tr]); pred=model.predict(X[te])
     repeated=np.isin(keys[te],keys[tr]); rec={"seed":seed,"task":task,"protocol":protocol,"representation":rep,
      "n_train":len(tr),"n_test":len(te),"split_hash":split_hash,"train_class_counts":dict(Counter(map(str,ytask[tr]))),
      "test_class_counts":dict(Counter(map(str,ytask[te]))),"raw_overlap_classes":overlap_raw,"tda_overlap_classes":overlap_tda,
      "repeated_test_count":int(repeated.sum()),"novel_test_count":int((~repeated).sum()),"metrics":score(ytask[te],pred,classes),
      "repeated_accuracy":accuracy_score(ytask[te][repeated],pred[repeated]) if repeated.any() else None,
      "novel_accuracy":accuracy_score(ytask[te][~repeated],pred[~repeated]) if (~repeated).any() else None,"runtime_seconds":time.time()-t}
     cell.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); records.append(rec); print(cell)
 summary=[]
 for task in tasks:
  for protocol in ['random_row','group_raw_payload','group_tda_vector']:
   for rep in ['raw','tda280']:
    rr=[r for r in records if r['task']==task and r['protocol']==protocol and r['representation']==rep]
    if not rr: continue
    row={"task":task,"protocol":protocol,"representation":rep,"n_seeds":len(rr)}
    for metric in ['accuracy','macro_f1','weighted_f1','balanced_accuracy']:
     vals=np.array([r['metrics'][metric] for r in rr]); row[metric+'_mean']=vals.mean(); row[metric+'_sd']=vals.std(ddof=0)
     row[metric+'_ci95_low']=vals.mean()-1.96*vals.std(ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else vals[0]
     row[metric+'_ci95_high']=vals.mean()+1.96*vals.std(ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else vals[0]
    summary.append(row)
 with open(CSVOUT,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
 result={"experiment":design['experiment'],"preregistration_hash":design['lock_hash'],"records":records,"summary":summary,
  "population":{"raw":equivalence_profile(Xraw,y),"tda280":equivalence_profile(Xtda,y),"excluded":[]},
  "notebook_saved":{"multiclass_tda_accuracy":.7922012893534456,"multiclass_raw_accuracy":.8420,
    "binary_tda_accuracy":.9580673425960696,"binary_raw_accuracy":.9920,"rf_random_state":"unset"},
  "runtime_seconds":time.time()-started,"resource":{"workers":WORKERS,"peak_rss_kib":peak_rss_kib(),
    "python":sys.version,"platform":platform.platform()}}
 OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(OUT)
if __name__=='__main__': main()
