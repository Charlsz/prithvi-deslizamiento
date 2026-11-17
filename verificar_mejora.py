"""Verificación independiente de selección, particiones y métricas de la ampliación."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,jaccard_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experimentos/mejora_rf/resultados'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol=json.loads((OUT/'protocolo.json').read_text(encoding='utf-8'))
    selected=json.loads((OUT/'seleccion_congelada.json').read_text(encoding='utf-8'))
    evaluation=json.loads((OUT/'evaluacion.json').read_text(encoding='utf-8'))
    assert evaluation['selection_sha256']==digest(OUT/'seleccion_congelada.json')
    assert evaluation['protocol_sha256']==digest(OUT/'protocolo.json')
    tuning=pd.read_csv(OUT/'validacion_umbrales.csv')
    best=tuning.sort_values(['f1','iou','threshold'],ascending=False).iloc[0]
    assert np.isclose(best.threshold,selected['threshold'])
    assert np.allclose(tuning.threshold,protocol['thresholds'])
    development=json.loads((OUT/'manifest_desarrollo.json').read_text(encoding='utf-8'))
    test=json.loads((OUT/'manifest_test.json').read_text(encoding='utf-8'))
    hashes={s:{e['image_sha256'] for e in development+test if e['split']==s} for s in protocol['samples']}
    for s in hashes:
        assert len(hashes[s])==len(protocol['samples'][s])
        for other in hashes:
            if s!=other:assert not hashes[s]&hashes[other]
    scores=pd.read_csv(OUT/'test_metricas_por_escena.csv')
    global_scores=pd.read_csv(OUT/'test_metricas_globales_por_clase.csv')
    summary=pd.read_csv(OUT/'test_metricas.csv').set_index('method')
    with np.load(OUT/'predicciones_test.npz') as a:
        y=a['labels'];v=y>=0
        assert len(y)==len(protocol['samples']['test'])
        for method in summary.index:
            p=a[method]
            assert p.shape==y.shape and np.isin(p[v],[0,1]).all()
            if method=='baseline':assert (p[v]==selected['majority']).all()
            else:
                t=.5 if method=='prithvi_rf_05' else selected['threshold']
                grid=(a['probabilities']>t).astype(int)
                np.testing.assert_array_equal(p,np.repeat(np.repeat(grid,16,1),16,2))
            for i,(truth,pred) in enumerate(zip(y,p)):
                valid=truth>=0;truth=truth[valid];pred=pred[valid]
                pr,re,f1,sup=precision_recall_fscore_support(truth,pred,labels=[0,1],zero_division=0)
                iou=jaccard_score(truth,pred,labels=[0,1],average=None,zero_division=0)
                for c in [0,1]:
                    predicted=(pred==c).sum()
                    if sup[c]==0:re[c]=np.nan
                    if sup[c]==0 and predicted==0:pr[c]=f1[c]=iou[c]=np.nan
                    recorded=scores[(scores.scene==i)&(scores.method==method)&(scores['class']==c)].iloc[0]
                    np.testing.assert_allclose(recorded[['precision','recall','f1','iou']].astype(float),[pr[c],re[c],f1[c],iou[c]],equal_nan=True,atol=1e-10)
                    assert recorded.support==sup[c]
            mat=confusion_matrix(y[v],p[v],labels=[0,1])
            np.testing.assert_array_equal(pd.read_csv(OUT/f'confusion_{method}.csv',index_col=0).values,mat)
            pr,re,f1,sup=precision_recall_fscore_support(y[v],p[v],labels=[0,1],zero_division=0)
            iou=jaccard_score(y[v],p[v],labels=[0,1],average=None,zero_division=0)
            recorded=global_scores[global_scores.method==method].set_index('class')
            for k,vals in [('precision',pr),('recall',re),('f1',f1),('iou',iou),('support',sup)]:
                np.testing.assert_allclose(recorded.loc[[0,1],k],vals,atol=1e-10)
            subset=scores[(scores.method==method)&(scores['class']==1)]
            np.testing.assert_allclose(summary.loc[method,['iou','f1','precision','recall']].astype(float),subset[['iou','f1','precision','recall']].mean(),atol=1e-10)
            assert summary.loc[method,'escenas_f1_definidas']==subset.f1.notna().sum()
    ext=json.loads((OUT/'evaluacion_extensiones.json').read_text(encoding='utf-8'))
    assert ext['selection_sha256']==digest(OUT/'seleccion_extensiones_congelada.json')
    assert ext['protocol_sha256']==digest(OUT/'protocolo_extensiones.json')
    ext_scores=pd.read_csv(OUT/'test_extensiones_por_escena.csv')
    ext_global=pd.read_csv(OUT/'test_extensiones_globales_por_clase.csv')
    ext_summary=pd.read_csv(OUT/'test_extensiones.csv').set_index('method')
    with np.load(OUT/'predicciones_extensiones.npz') as a:
        y=a['labels'];v=y>=0
        for method in ext_summary.index:
            p=a[method]
            np.testing.assert_array_equal(p,(a[method+'_probability']>ext['thresholds'][method]).astype(int))
            mat=confusion_matrix(y[v],p[v],labels=[0,1])
            np.testing.assert_array_equal(pd.read_csv(OUT/f'confusion_{method}.csv',index_col=0).values,mat)
            pr,re,f1,sup=precision_recall_fscore_support(y[v],p[v],labels=[0,1],zero_division=0)
            iou=jaccard_score(y[v],p[v],labels=[0,1],average=None,zero_division=0)
            recorded=ext_global[ext_global.method==method].set_index('class')
            for k,vals in [('precision',pr),('recall',re),('f1',f1),('iou',iou),('support',sup)]:
                np.testing.assert_allclose(recorded.loc[[0,1],k],vals,atol=1e-10)
            all_scores=[]
            for i,(truth,pred) in enumerate(zip(y,p)):
                valid=truth>=0;truth=truth[valid];pred=pred[valid]
                pr,re,f1,sup=precision_recall_fscore_support(truth,pred,labels=[0,1],zero_division=0)
                iou=jaccard_score(truth,pred,labels=[0,1],average=None,zero_division=0)
                for cl in [0,1]:
                    predicted=(pred==cl).sum()
                    if sup[cl]==0:re[cl]=np.nan
                    if sup[cl]==0 and predicted==0:pr[cl]=f1[cl]=iou[cl]=np.nan
                    row=ext_scores[(ext_scores.scene==i)&(ext_scores.method==method)&(ext_scores['class']==cl)].iloc[0]
                    np.testing.assert_allclose(row[['iou','f1','precision','recall']].astype(float),[iou[cl],f1[cl],pr[cl],re[cl]],equal_nan=True,atol=1e-10)
                all_scores.append([iou[1],f1[1],pr[1],re[1]])
            np.testing.assert_allclose(ext_summary.loc[method,['iou','f1','precision','recall']].astype(float),np.nanmean(all_scores,axis=0),atol=1e-10)
    result={'checks':'PASS','test_images':len(test),'selection_frozen':True,'no_identical_images_across_splits':True,'scope':'Verificación técnica; no garantiza independencia geográfica ni aprobación docente.'}
    (OUT/'verificacion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
