"""Compara cabezas solo con train/validation. Test sigue cerrado en esta etapa."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import joblib
import experimento_mejora as exp


def dense_features(raw,embedding,pca,mean,std,include_embedding=True):
    optical=exp.normalize(raw,mean,std)[:,0].transpose(1,2,0)
    if not include_embedding:return optical
    low=pca.transform(embedding.reshape(-1,embedding.shape[-1])).reshape(14,14,-1)
    # Interpolación solo de características continuas; nunca de referencias categóricas.
    up=torch.nn.functional.interpolate(torch.from_numpy(low.transpose(2,0,1).copy())[None],size=(224,224),mode='bilinear',align_corners=False)[0].numpy().transpose(1,2,0)
    return np.concatenate([optical,up],axis=-1)


def main():
    out=exp.OUT
    if (out/'test_metricas.csv').exists() or (out/'test_extensiones.csv').exists():raise RuntimeError('Test ya abierto: no se permite desarrollar más cabezas en esta corrida.')
    p=exp.protocol()
    choices=dict(candidates=['patch_occupancy_005','patch_occupancy_010','patch_occupancy_020','dense_prithvi_optical','dense_optical_only'],
        occupancy_rules='Positivo si fracción de píxeles válidos > 0.05, 0.10 o 0.20; mínimo 50% validez del parche.',
        dense='RF 100 árboles profundidad20, bandas ópticas normalizadas y PCA32 de embeddings interpolados; encoder congelado.',
        dense_sampling='Hasta 512 píxeles de cada clase por escena train, muestreo sin reemplazo, semilla42.',
        thresholds=p['thresholds'],metric=p['selection_metric'],
        status='Extensiones exploratorias; cambian objetivos o unidad de predicción del flujo guía. Conservar el piloto y solicitar aceptación docente para usarlas como método principal.')
    exp.save(out/'protocolo_extensiones.json',choices)
    stats=json.loads((out/'normalizacion.json').read_text(encoding='utf-8'))
    mean=np.array(stats['mean'],dtype=np.float32);std=np.array(stats['std'],dtype=np.float32)
    signature=__import__('hashlib').sha256(np.concatenate([mean,std]).tobytes()).hexdigest()[:16]
    cache={s:np.load(exp.WORK/'cache'/f'{s}_{signature}.npz') for s in ['train','validation']}
    train_f=cache['train']['features'];train_y=cache['train']['labels'];val_f=cache['validation']['features'];val_y=cache['validation']['labels']
    allrows=[];selection={}
    def choose(name,probabilities):
        results=[]
        for threshold in p['thresholds']:
            mm=pd.DataFrame([exp.score_scene(y,(prob>threshold).astype(int)) for y,prob in zip(val_y,probabilities)])
            results.append(dict(candidate=name,threshold=threshold,**{k:float(mm[k].mean()) for k in ['iou','f1','precision','recall']},defined_f1=int(mm.f1.notna().sum())))
        table=pd.DataFrame(results);best=table.sort_values(['f1','iou','threshold'],ascending=False).iloc[0].to_dict()
        selection[name]=best;allrows.extend(results)
        print('VALIDACIÓN',json.dumps(best,ensure_ascii=False),flush=True)
        return best
    for fraction in [.05,.1,.2]:
        name={.05:'patch_occupancy_005',.1:'patch_occupancy_010',.2:'patch_occupancy_020'}[fraction]
        labels=np.full((len(train_y),14,14),-1,dtype=int)
        for i,y in enumerate(train_y):
            for r in range(14):
                for c in range(14):
                    block=y[r*16:(r+1)*16,c*16:(c+1)*16];valid=block>=0
                    if valid.sum()>=128:labels[i,r,c]=int((block==1).sum()/valid.sum()>fraction)
        valid=labels>=0;head=RandomForestClassifier(**p['head']);head.fit(train_f[valid],labels[valid])
        pos=int(np.where(head.classes_==1)[0][0])
        probs=head.predict_proba(val_f.reshape(-1,val_f.shape[-1]))[:,pos].reshape(len(val_f),14,14)
        probs=np.repeat(np.repeat(probs,16,1),16,2)
        choose(name,probs);joblib.dump(head,exp.WORK/f'{name}.joblib')
    valid=exp.patch_labels(train_y[0])>=0
    train_tokens=np.concatenate([f[exp.patch_labels(y)>=0] for f,y in zip(train_f,train_y)])
    pca=PCA(n_components=32,random_state=42).fit(train_tokens)
    rng=np.random.default_rng(42);xx=[];targets=[]
    for i,(entry,embedding,y) in enumerate(zip(p['samples']['train'],train_f,train_y)):
        raw,_=exp.read_sample('deslizamientos',exp.WORK/'datos'/entry['image'],exp.WORK/'datos'/entry['label'])
        ff=dense_features(raw,embedding,pca,mean,std)
        for cl in [0,1]:
            indices=np.flatnonzero(y.ravel()==cl)
            take=rng.choice(indices,size=min(512,len(indices)),replace=False)
            xx.append(ff.reshape(-1,ff.shape[-1])[take]);targets.append(np.full(len(take),cl))
    xx=np.concatenate(xx);targets=np.concatenate(targets)
    print('Entrenamiento denso con píxeles reales:',xx.shape,np.unique(targets,return_counts=True),flush=True)
    validation_features=[]
    for entry,embedding in zip(p['samples']['validation'],val_f):
        raw,_=exp.read_sample('deslizamientos',exp.WORK/'datos'/entry['image'],exp.WORK/'datos'/entry['label'])
        validation_features.append(dense_features(raw,embedding,pca,mean,std))
    for name,channels in [('dense_prithvi_optical',slice(None)),('dense_optical_only',slice(0,6))]:
        head=RandomForestClassifier(n_estimators=100,max_depth=20,class_weight='balanced',random_state=42,n_jobs=2)
        head.fit(xx[:,channels],targets)
        probs=[]
        for ff in validation_features:
            reduced=ff[:,:,channels];probs.append(head.predict_proba(reduced.reshape(-1,reduced.shape[-1]))[:,1].reshape(224,224))
        choose(name,np.array(probs));joblib.dump(head,exp.WORK/f'{name}.joblib')
    joblib.dump(pca,exp.WORK/'pca_embeddings.joblib')
    pd.DataFrame(allrows).to_csv(out/'validacion_extensiones.csv',index=False)
    best=max([v for k,v in selection.items() if k!='dense_optical_only'],key=lambda r:(r['f1'],r['iou'],r['threshold']))
    # El control óptico se reporta aunque no gane: permite medir el aporte de Prithvi.
    exp.save(out/'seleccion_extensiones_congelada.json',dict(candidates=selection,selected=best,control=selection['dense_optical_only'],
        frozen_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),status=choices['status']))
    print('MEJOR CANDIDATO CON PRITHVI:',best,flush=True)


if __name__=='__main__':main()
