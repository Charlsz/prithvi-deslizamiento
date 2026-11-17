"""Evalúa una selección ya congelada. No entrena ni selecciona con test."""
import json,hashlib
import joblib
import numpy as np
import pandas as pd
from importlib.metadata import version
import experimento_mejora as exp
from desarrollar_cabezas import dense_features


def main():
    out=exp.OUT
    if (out/'test_extensiones.csv').exists():raise RuntimeError('Evaluación de extensiones ya registrada. No volver a seleccionar con este test.')
    path=out/'seleccion_extensiones_congelada.json'
    selection=json.loads(path.read_text(encoding='utf-8'))
    assert (out/'test_metricas.csv').exists(), 'Ejecutar primero el flujo de evaluación del RF mayoritario.'
    p=exp.protocol();stats=json.loads((out/'normalizacion.json').read_text(encoding='utf-8'))
    mean=np.array(stats['mean'],dtype=np.float32);std=np.array(stats['std'],dtype=np.float32)
    signature=hashlib.sha256(np.concatenate([mean,std]).tobytes()).hexdigest()[:16]
    cached=np.load(exp.WORK/'cache'/f'test_{signature}.npz');ff=cached['features'];yy=cached['labels']
    pca=joblib.load(exp.WORK/'pca_embeddings.joblib')
    chosen=selection['selected'];control=selection['control']
    best_patch=max((v for k,v in selection['candidates'].items() if k.startswith('patch_')),key=lambda r:(r['f1'],r['iou'],r['threshold']))
    # Selección de la variante por parche también procede exclusivamente de validación.
    candidates={chosen['candidate']:chosen,'dense_optical_only':control,best_patch['candidate']:best_patch}
    predictions={};probabilities={};scene=[];globalrows=[]
    for name,spec in candidates.items():
        head=joblib.load(exp.WORK/f'{name}.joblib');probs=[];preds=[];total=np.zeros((2,2),dtype=np.int64)
        for i,(entry,embedding,y) in enumerate(zip(p['samples']['test'],ff,yy)):
            if name.startswith('patch_'):
                prob=head.predict_proba(embedding.reshape(-1,embedding.shape[-1]))[:,1].reshape(14,14)
                prob=np.repeat(np.repeat(prob,16,0),16,1)
            else:
                raw,_=exp.read_sample('deslizamientos',exp.WORK/'datos'/entry['image'],exp.WORK/'datos'/entry['label'])
                f=dense_features(raw,embedding,pca,mean,std,include_embedding=(name!='dense_optical_only'))
                prob=head.predict_proba(f.reshape(-1,f.shape[-1]))[:,1].reshape(224,224)
            pred=(prob>spec['threshold']).astype(int);probs.append(prob);preds.append(pred)
            mat=exp.confusion(y,pred,[0,1]);total+=mat
            for r in exp.metrics(mat,[0,1]):scene.append(dict(scene=i,image=entry['image'],method=name,**r))
        predictions[name]=np.array(preds);probabilities[name+'_probability']=np.array(probs)
        for r in exp.metrics(total,[0,1]):globalrows.append(dict(method=name,**r))
        pd.DataFrame(total,index=[0,1],columns=[0,1]).to_csv(out/f'confusion_{name}.csv')
    np.savez_compressed(out/'predicciones_extensiones.npz',labels=yy,**predictions,**probabilities)
    frame=pd.DataFrame(scene);frame.to_csv(out/'test_extensiones_por_escena.csv',index=False)
    pd.DataFrame(globalrows).to_csv(out/'test_extensiones_globales_por_clase.csv',index=False)
    pos=frame[frame['class']==1]
    summary=pos.groupby('method')[['iou','f1','precision','recall']].mean().join(pos.groupby('method').f1.count().rename('escenas_f1_definidas'))
    summary.to_csv(out/'test_extensiones.csv')
    exp.save(out/'evaluacion_extensiones.json',dict(selected=chosen,thresholds={n:v['threshold'] for n,v in candidates.items()},
        selection_sha256=exp.sha256(path),protocol_sha256=exp.sha256(out/'protocolo_extensiones.json'),
        packages={n:version(n) for n in ['torch','timm','numpy','scikit-learn','pandas','h5py','huggingface_hub']},
        model_hash=exp.sha256(exp.ROOT/'taller_deslizamientos/modelo/Prithvi_100M.pt'),
        status=selection['status']))
    print('PRUEBA FINAL DE EXTENSIONES\n'+summary.to_string(),flush=True)


if __name__=='__main__':main()
