"""Evidence-locked, resumable Monkam representation audit."""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
from gtda.homology import CubicalPersistence
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from programs.monkam_representation import (SPECS, SUPPLIED_280_DEFINITION,
    equivalence_profile, feature_blocks, fit_shared, learned_state, reshape_payloads,
    stable_hash)
from programs.resource_usage import peak_rss_kib
from programs.run_monkam_126_fit_protocol_pilot import load_selected_payloads

RESULTS=ROOT/'results'
PRE=RESULTS/'monkam_representation_audit_preregistration.json'
OUT=RESULTS/'monkam_representation_audit.json'; SEED=42
PILOT_ROWS=[0,1,2,50138,50139,50140,79879,79880]

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()

def lock_document():
    inputs=['data/Payload_data_UNSW.csv','monkam_files/tda_280_x_y.xlsx',
            'monkam_files/tda_begnin_X_126.xlsx']
    doc={"experiment":"monkam_representation_audit","version":1,
      "dataset_and_input_hashes":{p:file_hash(ROOT/p) for p in inputs},
      "row_indices":PILOT_ROWS,"row_indices_hash":stable_hash(PILOT_ROWS),"seeds":[SEED],
      "parent_poison_relationships":"18 delivered examples from 11 filename-identified parents; diagnostic only",
      "preprocessing_and_models":{k:v.as_dict() for k,v in SPECS.items()} |
                                     {"supplied_280":SUPPLIED_280_DEFINITION},
      "primary_endpoints":["feature count and layout","1x1500 H1 degeneracy",
          "shared-fit learned states","raw/mask/diagram/vector equivalence"],
      "secondary_endpoints":["constant and zero-variance dimensions","label conflicts",
          "runtime and peak RSS"],
      "permitted_sensitivity":["completed separate-fit 18-poison pilot, read-only"],
      "confirmation_gates":{"finite":True,"shape":True,"hash_repeatability":True,
          "workbook_alignment_must_be_preverified":True},"effective_workers":1}
    doc['lock_hash']=stable_hash(doc); return doc

def branch_stage_profiles(X,spec,pipeline):
    images=reshape_payloads(X,spec); out={}
    for name,branch in pipeline.transformer_list:
        binary=branch.named_steps['binarizer'].transform(images)
        filtered=branch.named_steps['filtration'].transform(binary)
        diagrams=branch.named_steps['persistence'].transform(filtered)
        out[name]={"binary_mask":equivalence_profile(binary.reshape(len(X),-1)),
                   "diagram":equivalence_profile(diagrams.reshape(len(X),-1)),
                   "binary_hash":stable_hash(binary),"diagram_hash":stable_hash(diagrams)}
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--prepare-only',action='store_true'); args=ap.parse_args()
    RESULTS.mkdir(exist_ok=True); lock=lock_document()
    if args.prepare_only:
        PRE.write_text(json.dumps(lock,indent=2,sort_keys=True)+'\n'); print(PRE); return
    if not PRE.exists() or stable_hash(json.loads(PRE.read_text()))!=stable_hash(lock):
        raise RuntimeError('preregistration is missing or differs from reconstructed design')
    started=time.time(); X,y=load_selected_payloads(PILOT_ROWS); configs={}
    for name,spec in SPECS.items():
        vectors,pipeline=fit_shared(X,spec,n_jobs=1); layout=feature_blocks(spec)
        h1=[i for block in layout.values() for i in block['homology_indices']['h1']]
        configs[name]={"definition":spec.as_dict(),"observed_features":vectors.shape[1],
          "layout":layout,"learned_state":learned_state(pipeline),"feature_hash":stable_hash(vectors),
          "constant_dimensions":np.flatnonzero(np.ptp(vectors,axis=0)==0).tolist(),
          "zero_variance_dimensions":np.flatnonzero(np.var(vectors,axis=0)==0).tolist(),
          "h1_identically_constant":bool(np.all(np.ptp(vectors[:,h1],axis=0)==0)),
          "stages":branch_stage_profiles(X,spec,pipeline),
          "final_vector_equivalence":equivalence_profile(vectors,y)}
    inventory=json.loads((RESULTS/'monkam_materials_inventory.json').read_text())
    comparison=inventory['comparisons']; workbooks=comparison['xlsx_numeric_profiles']
    result={"experiment":lock['experiment'],"complete":True,"preregistration_hash":lock['lock_hash'],
      "frames":{"A":"legacy transposition collision provenance","B":"guaranteed-raw-changing four-family control",
                "delivered_poison":"18 examples from 11 parents, not the missing 1000"},
      "pilot":{"rows":PILOT_ROWS,"payload_hash":stable_hash(X),"raw_equivalence":equivalence_profile(X,y),
               "configurations":configs},"workbooks":workbooks,
      "workbook_alignment":comparison.get('raw_to_280_feature_collisions'),
      "supplied_280_definition":SUPPLIED_280_DEFINITION,
      "unrecoverable":["missing 1000 poison examples","attack generator/objective/seeds",
        "selected HDBSCAN parameters","selected OPTICS parameters","selected Mean Shift parameters",
        "feature-reduction configuration","unseeded 10000-row membership underlying supplied 126 workbook"],
      "historical_separate_fit_reference":"results/monkam_126_fit_protocol_pilot_seed60.json",
      "runtime_seconds":time.time()-started,"resource":{"workers":1,
        "peak_rss_kib":peak_rss_kib(),
        "python":sys.version,"platform":platform.platform()}}
    OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(OUT)
if __name__=='__main__': main()
