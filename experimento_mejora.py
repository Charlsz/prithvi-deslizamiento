"""Ampliación del piloto: mismo encoder congelado, etiquetas mayoritarias y RF.

Desarrollo y evaluación separados. Selección de escenas previa a leer etiquetas.
No reemplaza el piloto docente ni acredita aprobación de su contrato de entrada.
"""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
import joblib
from huggingface_hub.utils import logging as hf_logging
hf_logging.set_verbosity_error()

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'taller_deslizamientos'))
from prithvi_taller import read_sample, train_normalization, normalize, load_prithvi, features, patch_labels, confusion, metrics, sha256

WORK=ROOT/'experimentos'/'mejora_rf'
OUT=WORK/'resultados'
REV='8508a55feceb5653a085d6be64ef9217710f7c00'
REPO='sasmit545/Landslide4sense'


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def protocol():
    path=OUT/'protocolo.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    from huggingface_hub import HfApi
    files=HfApi().list_repo_files(REPO,repo_type='dataset',revision=REV)
    rng=np.random.default_rng(20261117)
    selected={}
    excluded={'train':{'image_1.h5','image_10.h5','image_100.h5','image_1000.h5'},'test':{'image_1.h5','image_10.h5'},'validation':set()}
    for split,n in [('train',96),('validation',24),('test',32)]:
        images=sorted(f for f in files if f.startswith(f'images/{split}/') and f.endswith('.h5') and Path(f).name not in excluded[split])
        selected[split]=[]
        for i in sorted(rng.choice(len(images),size=n,replace=False).tolist()):
            image=images[i];mask=f'annotations/{split}/mask_{Path(image).stem.split("_")[-1]}.h5'
            assert mask in files
            selected[split].append({'image':image,'label':mask})
    p=dict(seed=20261117,repo=REPO,revision=REV,selection='Muestreo uniforme de nombres, sin leer imágenes ni etiquetas; excluye las seis escenas previas.',
        samples=selected,head={'n_estimators':80,'max_depth':10,'class_weight':'balanced','random_state':42,'n_jobs':2},
        patch_target='Mayoría de píxeles válidos; empate a fondo; mínimo 50% de validez; parche16',
        thresholds=[.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75],
        selection_metric='F1 de clase positiva, media por escena de validación; desempate por IoU y luego mayor umbral',
        evaluation='Una sola evaluación de test después de guardar el umbral. No cambiar configuración después de evaluar.',
        teacher_status='Ampliación complementaria pendiente de aceptación del tamaño de subconjunto y confirmación de canales, escala, normalización y pesos',
        frozen_at_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    save(path,p);return p


def download(p,split):
    from huggingface_hub import hf_hub_download
    rows=p['samples'][split]
    names=[e[k] for e in rows for k in ['image','label']]
    def get(name):
        return hf_hub_download(REPO,filename=name,repo_type='dataset',revision=REV,local_dir=str(WORK/'datos'))
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(get,names))
    return [(WORK/'datos'/e['image'],WORK/'datos'/e['label']) for e in rows]


def extract(model,device,samples,split,mean,std):
    cache=WORK/'cache';cache.mkdir(exist_ok=True)
    signature=hashlib.sha256(np.concatenate([mean,std]).tobytes()).hexdigest()[:16]
    path=cache/f'{split}_{signature}.npz'
    if path.exists():
        a=np.load(path);return a['features'],a['labels']
    ff=[];yy=[]
    for i,(x,y) in enumerate(samples):
        ff.append(features(model,device,normalize(x,mean,std)));yy.append(y)
        if (i+1)%8==0:print(f'{split}: embeddings {i+1}/{len(samples)}',flush=True)
    ff=np.array(ff);yy=np.array(yy)
    np.savez_compressed(path,features=ff,labels=yy)
    return ff,yy


def score_scene(y,pred):
    return metrics(confusion(y,pred,[0,1]),[0,1])[1]


def develop(p):
    if (OUT/'test_metricas.csv').exists():raise RuntimeError('Test ya evaluado. No ajustar el desarrollo después de ver test.')
    data={s:download(p,s) for s in ['train','validation']}
    samples={s:[read_sample('deslizamientos',im,lab) for im,lab in rows] for s,rows in data.items()}
    mean,std=train_normalization(samples['train'])
    save(OUT/'normalizacion.json',dict(mean=mean.tolist(),std=std.tolist()))
    model,device,info=load_prithvi(ROOT/'taller_deslizamientos/modelo',num_frames=1)
    train_f,train_y=extract(model,device,samples['train'],'train',mean,std)
    targets=np.array([patch_labels(y) for y in train_y]);valid=targets>=0
    unique,counts=np.unique(targets[valid],return_counts=True)
    print('Clases de parches de entrenamiento:',dict(zip(unique.tolist(),counts.tolist())),flush=True)
    if set(unique)!={0,1}:raise RuntimeError('Falta una clase; no se evalúa test.')
    head=RandomForestClassifier(**p['head']);head.fit(train_f[valid],targets[valid])
    val_f,val_y=extract(model,device,samples['validation'],'validation',mean,std)
    pos=int(np.where(head.classes_==1)[0][0])
    probs=head.predict_proba(val_f.reshape(-1,train_f.shape[-1]))[:,pos].reshape(len(val_f),14,14)
    rows=[]
    for threshold in p['thresholds']:
        mm=[]
        for y,prob in zip(val_y,probs):
            pred=np.repeat(np.repeat((prob>threshold).astype(int),16,0),16,1)
            mm.append(score_scene(y,pred))
        frame=pd.DataFrame(mm)
        rows.append(dict(threshold=threshold,**{k:float(frame[k].mean()) for k in ['iou','f1','precision','recall']},defined_f1=int(frame.f1.notna().sum())))
    table=pd.DataFrame(rows);table.to_csv(OUT/'validacion_umbrales.csv',index=False)
    best=table.sort_values(['f1','iou','threshold'],ascending=False).iloc[0]
    all_y=train_y[train_y>=0];vc,cc=np.unique(all_y,return_counts=True);majority=int(vc[np.argmax(cc)])
    selection=dict(threshold=float(best.threshold),validation=best.to_dict(),majority=majority,
        train_patch_classes={str(k):int(v) for k,v in zip(unique,counts)},
        train_pixel_classes={str(k):int(v) for k,v in zip(vc,cc)},model=info,
        configuration_frozen_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    joblib.dump(head,WORK/'cabeza_rf.joblib')
    save(OUT/'seleccion_congelada.json',selection)
    manifest=[]
    for s,rows0 in data.items():
        for im,lab in rows0:manifest.append(dict(split=s,image=im.relative_to(WORK/'datos').as_posix(),label=lab.relative_to(WORK/'datos').as_posix(),image_sha256=sha256(im),label_sha256=sha256(lab)))
    save(OUT/'manifest_desarrollo.json',manifest)
    print('Selección final SOLO de validación:',json.dumps(selection,ensure_ascii=False),flush=True)


def evaluate(p):
    if (OUT/'test_metricas.csv').exists():raise RuntimeError('La evaluación final ya existe. No repetir selección después de ver test.')
    selection_path=OUT/'seleccion_congelada.json'
    selected=json.loads(selection_path.read_text(encoding='utf-8'))
    selection_hash=sha256(selection_path)
    train_manifest=json.loads((OUT/'manifest_desarrollo.json').read_text(encoding='utf-8'))
    known={e['image_sha256'] for e in train_manifest}
    rows=download(p,'test')
    test_manifest=[]
    for im,lab in rows:
        digest=sha256(im)
        assert digest not in known,'Archivo duplicado con desarrollo'
        test_manifest.append(dict(split='test',image=im.relative_to(WORK/'datos').as_posix(),label=lab.relative_to(WORK/'datos').as_posix(),image_sha256=digest,label_sha256=sha256(lab)))
    save(OUT/'manifest_test.json',test_manifest)
    samples=[read_sample('deslizamientos',im,lab) for im,lab in rows]
    stats=json.loads((OUT/'normalizacion.json').read_text(encoding='utf-8'))
    model,device,info=load_prithvi(ROOT/'taller_deslizamientos/modelo',num_frames=1)
    ff,yy=extract(model,device,samples,'test',np.array(stats['mean'],dtype=np.float32),np.array(stats['std'],dtype=np.float32))
    head=joblib.load(WORK/'cabeza_rf.joblib');pos=int(np.where(head.classes_==1)[0][0])
    probs=head.predict_proba(ff.reshape(-1,ff.shape[-1]))[:,pos].reshape(len(ff),14,14)
    outputs={};scene=[];global_rows=[]
    for method,threshold in [('prithvi_rf_05',.5),('prithvi_rf_validado',selected['threshold']),('baseline',None)]:
        total=np.zeros((2,2),dtype=np.int64);predictions=[]
        for i,(y,prob) in enumerate(zip(yy,probs)):
            pred=np.full(y.shape,selected['majority']) if threshold is None else np.repeat(np.repeat((prob>threshold).astype(int),16,0),16,1)
            predictions.append(pred);mat=confusion(y,pred,[0,1]);total+=mat
            for r in metrics(mat,[0,1]):scene.append(dict(scene=i,image=rows[i][0].name,method=method,**r))
        outputs[method]=np.array(predictions)
        for r in metrics(total,[0,1]):global_rows.append(dict(method=method,**r))
        pd.DataFrame(total,index=[0,1],columns=[0,1]).to_csv(OUT/f'confusion_{method}.csv')
    np.savez_compressed(OUT/'predicciones_test.npz',labels=yy,probabilities=probs,**outputs)
    frame=pd.DataFrame(scene);frame.to_csv(OUT/'test_metricas_por_escena.csv',index=False)
    pd.DataFrame(global_rows).to_csv(OUT/'test_metricas_globales_por_clase.csv',index=False)
    positive=frame[frame['class']==1]
    mean=positive.groupby('method')[['iou','f1','precision','recall']].mean()
    mean=mean.join(positive.groupby('method').f1.count().rename('escenas_f1_definidas'))
    mean.to_csv(OUT/'test_metricas.csv')
    save(OUT/'evaluacion.json',dict(selection_sha256=selection_hash,protocol_sha256=sha256(OUT/'protocolo.json'),
        split_sizes={s:len(v) for s,v in p['samples'].items()},threshold=selected['threshold'],
        teacher_status=p['teacher_status'],model=info,
        limitations='Independencia por archivo y partición del espejo; sin metadatos suficientes para acreditar independencia por evento o región. Nuevas escenas de test respecto al piloto, sin garantía de dominio independiente.'))
    print('TEST FINAL - media por escena\n',mean.to_string(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['develop','evaluate']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    p=protocol()
    develop(p) if args.phase=='develop' else evaluate(p)
