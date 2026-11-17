"""Comprueba coherencia numérica y ejecución; no certifica aprobación docente."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, jaccard_score
import nbformat

ROOT = Path(__file__).resolve().parent


def main():
    out = ROOT / 'entregables'
    nb = nbformat.read(ROOT/'Grupo_2_Trabajo_Deslizamientos.ipynb',as_version=4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == 'code']
    counts = [c.execution_count for c in code]
    assert all(isinstance(n,int) for n in counts), 'Hay celdas sin ejecutar'
    assert counts == list(range(1,len(code)+1)), 'La ejecución no es secuencial'
    assert not any(o.output_type == 'error' for c in code for o in c.outputs)
    required=['metricas.csv','metricas_por_clase.csv','metricas_por_escena.csv',
              'metricas_media_por_escena.csv','distribucion_train.csv','procedencia.json',
              'normalizacion.json','manifest_datos.json','diagnostico.json','diagnostico_parches.csv',
              'inventario_muestras.csv','casos_errores.csv','01_exploracion.png',
              '03_comparacion_metricas.png']
    for name in required:
        assert (out/name).is_file() and (out/name).stat().st_size>0, name
    for p in out.glob('*.json'):json.loads(p.read_text(encoding='utf-8'))
    inv=pd.read_csv(out/'inventario_muestras.csv')
    train=inv[inv.split=='train'];test=inv[inv.split=='test']
    assert len(train)==4 and len(test)==2
    distribution=pd.read_csv(out/'distribucion_train.csv').set_index('clase')
    assert distribution.loc[1,'pixeles']==train.positive.sum()
    assert distribution.pixeles.sum()==train.valid.sum()
    majority=int(distribution.pixeles.idxmax())
    scene_metrics=pd.read_csv(out/'metricas_por_escena.csv')
    global_metrics=pd.read_csv(out/'metricas_por_clase.csv')
    summary=pd.read_csv(out/'metricas.csv').set_index('method')
    mean=pd.read_csv(out/'metricas_media_por_escena.csv').set_index('method')
    seen=[]
    for method,field in [('prithvi_rf','prediction'),('baseline','baseline')]:
        truths=[];predictions=[];positive_rows=[]
        for i in range(len(test)):
            for pattern in ['02_resultado_{}.png','04_trazabilidad_{}.png','05_detalle_FN_{}.png']:
                assert (out/pattern.format(i)).is_file()
            with np.load(out/f'escena_{i}.npz') as a:
                y=a['label'];p=a[field];v=y>=0
                assert y.shape==p.shape==(224,224)
                assert v.sum()==int(test.iloc[i].valid)
                assert np.isin(p[v],[0,1]).all()
                if method=='baseline':assert (p[v]==majority).all()
                yt,yp=y[v],p[v]
                truths.append(yt);predictions.append(yp)
                pr,re,f1,support=precision_recall_fscore_support(yt,yp,labels=[0,1],zero_division=0)
                iou=jaccard_score(yt,yp,labels=[0,1],average=None,zero_division=0)
                recorded=scene_metrics[(scene_metrics.scene==i)&(scene_metrics.method==method)].set_index('class')
                for k,values in [('precision',pr),('recall',re),('f1',f1),('iou',iou),('support',support)]:
                    np.testing.assert_allclose(recorded.loc[[0,1],k],values,rtol=1e-9,atol=1e-12)
                positive_rows.append([iou[1],f1[1],pr[1],re[1]])
        yt=np.concatenate(truths);yp=np.concatenate(predictions)
        mat=confusion_matrix(yt,yp,labels=[0,1])
        np.testing.assert_array_equal(pd.read_csv(out/f'confusion_{method}.csv',index_col=0).values,mat)
        pr,re,f1,support=precision_recall_fscore_support(yt,yp,labels=[0,1],zero_division=0)
        iou=jaccard_score(yt,yp,labels=[0,1],average=None,zero_division=0)
        recorded=global_metrics[global_metrics.method==method].set_index('class')
        for k,values in [('precision',pr),('recall',re),('f1',f1),('iou',iou),('support',support)]:
            np.testing.assert_allclose(recorded.loc[[0,1],k],values,rtol=1e-9,atol=1e-12)
        np.testing.assert_allclose(summary.loc[method,['iou_positive','f1_positive','precision','recall']].astype(float),[iou[1],f1[1],pr[1],re[1]])
        np.testing.assert_allclose(mean.loc[method,['iou','f1','precision','recall']].astype(float),np.mean(positive_rows,axis=0))
        assert (mean.loc[method,[f'escenas_definidas_{k}' for k in ['iou','f1','precision','recall']]]==2).all()
        seen.append(dict(method=method,confusion=mat.tolist(),valid_pixels=len(yt)))
    manifest=json.loads((out/'manifest_datos.json').read_text(encoding='utf-8'))
    hashes={s:{e['image_sha256'] for e in manifest['samples'] if e['split']==s} for s in ['train','test']}
    assert not hashes['train'] & hashes['test']
    patches=pd.read_csv(out/'diagnostico_parches.csv')
    targets=patches[(patches.split=='train')&(patches.target>=0)]
    assert len(targets)==256 and set(targets.target)=={0}
    assert np.isclose(targets.positive_fraction.max(),.3984375)
    # Casos límite distintos al piloto: ignorados, ausencia y precisión sin predicción.
    import importlib.util
    spec=importlib.util.spec_from_file_location('verify_module',ROOT/'taller_deslizamientos/prithvi_taller.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.confusion(np.array([[1,-1]]),np.array([[0,1]]),[0,1]).sum()==1
    absent=module.metrics(np.array([[4,0],[0,0]]),[0,1])[1]
    assert all(np.isnan(absent[k]) for k in ['iou','f1','precision','recall'])
    missing_pred=module.metrics(np.array([[4,0],[2,0]]),[0,1])[1]
    assert all(missing_pred[k]==0 for k in ['iou','f1','precision','recall'])
    result=dict(technical_checks='PASS',executed_code_cells=len(code),numeric_checks=seen,
                teacher_confirmation='PENDIENTE: canales, escala, normalización y pesos',
                publication='No comprobada por este verificador',
                scope='Coherencia técnica, no certificación de nota ni aprobación del profesor')
    (out/'revision_entrega.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
