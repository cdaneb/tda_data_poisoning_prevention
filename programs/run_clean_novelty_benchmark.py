"""Resumable clean-calibrated novelty benchmark on Phase Q Frame B."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,platform,sys,time
from pathlib import Path
import numpy as np
import sklearn, scipy, gtda
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from programs.data_loader import load_unsw
from programs.monkam_representation import stable_hash
from programs.novelty_detectors import detector_factories,fit_calibrate_evaluate
from programs.phase_q_attacks import SUPPORTED_FAMILIES
from programs.phase_q_pipeline import CONTROL_THRESHOLD,THRESHOLD_STACK,extract_multithreshold_features
from programs.resource_usage import peak_rss_kib
from programs.run_test_b_capture import SEEDS,subsample_for_seed

RESULTS=ROOT/'results'; CELLS=RESULTS/'clean_novelty_cells'; CACHE=ROOT/'.step3_cache'
PRE=RESULTS/'clean_novelty_benchmark_preregistration.json'; OUT=RESULTS/'clean_novelty_benchmark.json'
CSVOUT=RESULTS/'clean_novelty_benchmark_summary.csv'; BUDGETS=[.001,.005,.01,.02,.05]; WORKERS=8
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def lock():
 d={"experiment":"phase_q_clean_calibrated_novelty_benchmark","frame":"B",
  "dataset_hash":sha(ROOT/'data/Payload_data_UNSW.csv'),"source_artifact_hash":sha(RESULTS/'phase_q_r1_multithreshold_capture.json'),
  "seeds":list(SEEDS),"families":list(SUPPORTED_FAMILIES),"sample":{"unmodified":5000,"poison":500},
  "parent_poison_relationships":"attack logs identify each appended poison source index",
  "representations":{"control":{"thresholds":[.4],"features":60},
    "stack":{"thresholds":list(THRESHOLD_STACK),"features":540,"scale":"1/sqrt(9)"}},
  "clean_split":{"group":"exact raw payload hash","detector_train":.6,"calibration":.2,"heldout_evaluation":.2},
  "preprocessing":"StandardScaler fitted on detector-training clean only; constants recorded",
  "detectors":{"knn_distance":{"k":10},"empirical_knn":{"k":10,"calibration":"empirical clean quantile"},
    "lof_novelty":{"n_neighbors":20},"isolation_forest":{"n_estimators":200},
    "one_class_svm":{"kernel":"rbf","gamma":"scale","nu":.01}},
  "threshold_budgets":BUDGETS,"threshold_source":"calibration-clean only",
  "primary_endpoint":"poison capture at each fixed clean-removal budget",
  "secondary_endpoints":["heldout clean removal","precision","AUROC","AUPRC","score distributions",
    "family/seed variation","exact clean-vector sharing","OPTICS clean/poison noise","paired OPTICS differences"],
  "permitted_sensitivity":[],"confirmation_gates":{"seed42_mechanism":True,"raw_noops":0,
    "finite_scores":True,"cache_hash_valid":True,"all_detectors_required":True},
  "parallelism":{"level":"detector internal only","effective_workers":WORKERS}}
 d['lock_hash']=stable_hash(d); return d
def row_hashes(X): return np.asarray([hashlib.sha256(np.ascontiguousarray(r).view(np.uint8)).hexdigest() for r in X])
def split_clean(X,seed):
 groups=row_hashes(X); idx=np.arange(len(X)); train,rest=next(GroupShuffleSplit(1,test_size=.4,random_state=seed).split(idx,groups=groups))
 calrel,evalrel=next(GroupShuffleSplit(1,test_size=.5,random_state=seed+10000).split(rest,groups=groups[rest]))
 cal,ev=rest[calrel],rest[evalrel]
 if set(groups[train])&set(groups[cal]) or set(groups[train])&set(groups[ev]) or set(groups[cal])&set(groups[ev]):
  raise AssertionError('raw payload group crossed clean partitions')
 return train,cal,ev,groups
def fitted_state(pipelines):
 out={}
 for threshold,union in pipelines.items():
  branches=[]
  for _,branch in union.transformer_list:
   steps=list(branch.named_steps.values()); b=next(x for x in steps if hasattr(x,'max_value_')); s=next(x for x in steps if hasattr(x,'scale_'))
   branches.append({"binarizer_max":float(b.max_value_),"effective_cut":float(b.max_value_*b.threshold),"scaler_scale":float(s.scale_)})
  out[str(threshold)]=branches
 return out
def cache_cell(Xfull,yfull,family,seed):
 CACHE.mkdir(exist_ok=True); path=CACHE/f'{family}_seed{seed}.npz'; meta=CACHE/f'{family}_seed{seed}.json'
 X,y=subsample_for_seed(Xfull,yfull,seed); fn,kwargs=SUPPORTED_FAMILIES[family]
 Xc,_,poisoned,log=fn(X,y,poison_rate=.1,random_state=seed,**kwargs)
 rawhash=stable_hash(Xc)
 if path.exists() and meta.exists():
  m=json.load(open(meta)); z=np.load(path)
  if m['combined_raw_hash']!=rawhash or m['control_hash']!=stable_hash(z['control']) or m['stack_hash']!=stable_hash(z['stack']):
   raise RuntimeError('cache content/hash mismatch')
  return Xc,poisoned,log,z['control'],z['stack'],m
 t=time.time(); stack,blocks,pipelines=extract_multithreshold_features(Xc,return_blocks=True); control=blocks[CONTROL_THRESHOLD]
 np.savez_compressed(path,control=control,stack=stack)
 m={"family":family,"seed":seed,"combined_raw_hash":rawhash,"poison_mask_hash":stable_hash(poisoned),
  "parent_indices":[x['target_index'] for x in log],"parent_relationship_hash":stable_hash([x['target_index'] for x in log]),
  "control_hash":stable_hash(control),"stack_hash":stable_hash(stack),"fitted_state":fitted_state(pipelines),
  "library_versions":{"numpy":np.__version__},"feature_runtime_seconds":time.time()-t,
  "resource":{"workers":WORKERS,"peak_rss_kib":peak_rss_kib()}}
 meta.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n'); return Xc,poisoned,log,control,stack,m
def run_cell(Xfull,yfull,family,seed,rep,features):
 cell=CELLS/f'{family}_{rep}_seed{seed}.json'
 if cell.exists(): return json.load(open(cell))
 Xc,poisoned,log,_,_,meta=cache_cell(Xfull,yfull,family,seed); nclean=5000
 train,cal,ev,groups=split_clean(Xc[:nclean],seed); poison_idx=np.arange(nclean,len(Xc))
 scaler=StandardScaler().fit(features[train]); Ztrain=scaler.transform(features[train]); Zcal=scaler.transform(features[cal]); Zev=scaler.transform(features[ev]); Zp=scaler.transform(features[poison_idx])
 detectors={}; labels=np.r_[np.zeros(len(Zev)),np.ones(len(Zp))]
 for name,factory in detector_factories(seed,WORKERS).items():
  started=time.time(); _,metrics,scores=fit_calibrate_evaluate(factory,Ztrain,Zcal,Zev,Zp,BUDGETS); cal_scores,clean_scores,poison_scores=scores
  combined=np.r_[clean_scores,poison_scores]
  detectors[name]={"budgets":metrics,"auroc":roc_auc_score(labels,combined),"auprc":average_precision_score(labels,combined),
   "score_distribution":{"calibration":{"mean":float(np.mean(cal_scores)),"sd":float(np.std(cal_scores)),"quantiles":np.quantile(cal_scores,[0,.25,.5,.75,1]).tolist()},
    "clean_eval":{"mean":float(np.mean(clean_scores)),"sd":float(np.std(clean_scores)),"quantiles":np.quantile(clean_scores,[0,.25,.5,.75,1]).tolist()},
    "poison":{"mean":float(np.mean(poison_scores)),"sd":float(np.std(poison_scores)),"quantiles":np.quantile(poison_scores,[0,.25,.5,.75,1]).tolist()}},
   "runtime_seconds":time.time()-started}
 sharing=float(np.mean(np.isin(row_hashes(features[nclean:]),row_hashes(features[:nclean]))))
 optics=json.load(open(RESULTS/'phase_q_r1_multithreshold_capture.json'))['runs'][family][str(seed)]
 arm=optics['control' if rep=='control' else 'repair']; curve=arm['removal_curve'][0]
 rec={"family":family,"seed":seed,"representation":rep,"feature_hash":stable_hash(features),"cache_metadata_hash":stable_hash(meta),
  "split":{"train":train.tolist(),"calibration":cal.tolist(),"evaluation":ev.tolist(),"hash":stable_hash({'t':train.tolist(),'c':cal.tolist(),'e':ev.tolist()}),
           "counts":[len(train),len(cal),len(ev)],"raw_group_overlap":0},
  "preprocessing":{"mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist(),"var":scaler.var_.tolist(),
    "constant_dimensions":np.flatnonzero(scaler.var_==0).tolist(),"state_hash":stable_hash(np.c_[scaler.mean_,scaler.scale_,scaler.var_])},
  "exact_poison_vector_sharing_clean":sharing,"detectors":detectors,
  "optics":{"poison_noise":curve['poison_unclustered_fraction'],"clean_noise":curve['clean_unclustered_fraction'],
            "exact_purity_capture":curve['poison_removal_rate']}}
 cell.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(cell); return rec
def merge(design):
 records=[json.load(open(p)) for p in sorted(CELLS.glob('*.json'))]; rows=[]
 for rep in ['control','stack']:
  for family in SUPPORTED_FAMILIES:
   for detector in detector_factories(0):
    rr=[r for r in records if r['representation']==rep and r['family']==family]
    if not rr: continue
    for budget in BUDGETS:
     vals=np.array([r['detectors'][detector]['budgets'][str(budget)]['poison_capture'] for r in rr]); clean=np.array([r['detectors'][detector]['budgets'][str(budget)]['clean_removal_rate'] for r in rr])
     row={"representation":rep,"family":family,"detector":detector,"budget":budget,"n_seeds":len(vals),
      "poison_capture_mean":vals.mean(),"poison_capture_sd":vals.std(ddof=0),"clean_removal_mean":clean.mean(),
      "ci95_low":vals.mean()-1.96*vals.std(ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else vals[0],
      "ci95_high":vals.mean()+1.96*vals.std(ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else vals[0]}
     rows.append(row)
 if rows:
  with open(CSVOUT,'w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 result={"experiment":design['experiment'],"preregistration_hash":design['lock_hash'],"records":records,"summary":rows,
  "complete":len(records)==len(SEEDS)*len(SUPPORTED_FAMILIES)*2,
  "library_versions":{"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,
    "scikit_learn":sklearn.__version__,"giotto_tda":gtda.__version__},
  "runtime":{"feature_seconds":sum(json.load(open(p))['feature_runtime_seconds'] for p in CACHE.glob('*.json')),
    "detector_seconds":sum(d['runtime_seconds'] for r in records for d in r['detectors'].values()),
    "workers":WORKERS,"platform":platform.platform(),"peak_rss_kib":peak_rss_kib()}}
 OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); return result
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--prepare-only',action='store_true'); ap.add_argument('--family',default='all',choices=list(SUPPORTED_FAMILIES)+['all']); ap.add_argument('--seed',default='all',choices=[str(s) for s in SEEDS]+['all']); a=ap.parse_args()
 design=lock(); RESULTS.mkdir(exist_ok=True); CELLS.mkdir(exist_ok=True)
 if a.prepare_only: PRE.write_text(json.dumps(design,indent=2,sort_keys=True)+'\n'); print(PRE); return
 if not PRE.exists() or stable_hash(json.load(open(PRE)))!=stable_hash(design): raise RuntimeError('preregistration mismatch')
 families=list(SUPPORTED_FAMILIES) if a.family=='all' else [a.family]; seeds=list(SEEDS) if a.seed=='all' else [int(a.seed)]
 Xfull,yfull=load_unsw()
 for family in families:
  for seed in seeds:
   Xc,poisoned,log,control,stack,meta=cache_cell(Xfull,yfull,family,seed)
   if any(x['raw_noop'] for x in log): raise AssertionError('raw no-op')
   run_cell(Xfull,yfull,family,seed,'control',control); run_cell(Xfull,yfull,family,seed,'stack',stack)
 merge(design); print(OUT)
if __name__=='__main__': main()
