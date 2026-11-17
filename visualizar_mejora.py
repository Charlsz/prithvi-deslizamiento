"""Figuras del test completo y tres casos diagnósticos, sin seleccionar el modelo."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import experimento_mejora as exp


def main():
    out=exp.OUT;p=exp.protocol()
    selected=json.loads((out/'seleccion_extensiones_congelada.json').read_text(encoding='utf-8'))['selected']['candidate']
    means=pd.concat([pd.read_csv(out/'test_metricas.csv'),pd.read_csv(out/'test_extensiones.csv')],ignore_index=True)
    glob=pd.concat([pd.read_csv(out/'test_metricas_globales_por_clase.csv'),pd.read_csv(out/'test_extensiones_globales_por_clase.csv')],ignore_index=True)
    pos=glob[glob['class']==1][['method','iou','f1','support']].rename(columns={'iou':'iou_global','f1':'f1_global'})
    comparison=means.merge(pos,on='method');comparison.to_csv(out/'comparacion_final.csv',index=False)
    names={'baseline':'Constante 0','prithvi_rf_05':'Mayoría RF 0.5','prithvi_rf_validado':'Mayoría RF validado','patch_occupancy_010':'Parche ocupación 10%','dense_prithvi_optical':'Prithvi + óptico por píxel','dense_optical_only':'Solo óptico por píxel'}
    fig,ax=plt.subplots(figsize=(10,4.8))
    x=np.arange(len(comparison));w=.35
    ax.bar(x-w/2,comparison.iou,w,label='IoU media por escena',color='#176a87')
    ax.bar(x+w/2,comparison.f1,w,label='F1 media por escena',color='#b63c78')
    ax.set_xticks(x,[names.get(m,m).replace(' ','\n',1) for m in comparison.method],fontsize=8)
    ax.set_ylim(0,1);ax.set_ylabel('Clase deslizamiento');ax.set_title('Prueba nueva completa | 32 escenas | selección solo con validación')
    for container in ax.containers:ax.bar_label(container,fmt='%.3f',padding=3,fontsize=8)
    ax.legend(loc='upper left',frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(out/'comparacion_final.png',dpi=180);plt.close(fig)
    data=np.load(out/'predicciones_extensiones.npz');yy=data['labels'];pred=data[selected]
    counts=[]
    for i,(y,pr) in enumerate(zip(yy,pred)):
        counts.append(dict(scene=i,image=p['samples']['test'][i]['image'],tp=int(((y==1)&(pr==1)).sum()),fp=int(((y==0)&(pr==1)).sum()),fn=int(((y==1)&(pr==0)).sum())))
    cases=pd.DataFrame(counts);cases.to_csv(out/'casos_test.csv',index=False)
    chosen={}
    shown=set()
    for typ,key in [('aciertos','tp'),('falsos_positivos','fp'),('falsos_negativos','fn')]:
        row=cases[~cases.scene.isin(shown)].sort_values([key,'scene'],ascending=[False,True]).iloc[0];i=int(row.scene);entry=p['samples']['test'][i]
        shown.add(i)
        chosen[typ]=dict(scene=i,image=entry['image'],criterion=f'Mayor número de {key.upper()} entre escenas aún no mostradas; desempate por índice. Selección diagnóstica posterior, sin ajustar el modelo.')
        raw,y=exp.read_sample('deslizamientos',exp.WORK/'datos'/entry['image'],exp.WORK/'datos'/entry['label'])
        rgb=raw[[2,1,0],0,:128,:128].transpose(1,2,0).copy()
        for c in range(3):
            lo,hi=np.percentile(rgb[:,:,c],[2,98]);rgb[:,:,c]=np.clip((rgb[:,:,c]-lo)/max(hi-lo,1e-6),0,1)
        pr=pred[i];err=np.zeros(y.shape,dtype=int)
        err[(y==1)&(pr==0)]=1;err[(y==0)&(pr==1)]=2;err[(y==1)&(pr==1)]=3
        fig,axes=plt.subplots(1,4,figsize=(13,4.1))
        axes[0].imshow(rgb,interpolation='nearest');axes[0].set_title('RGB observado')
        cm=ListedColormap(['#e9eef3','#b63c78'])
        axes[1].imshow(y[:128,:128],cmap=cm,vmin=0,vmax=1,interpolation='nearest');axes[1].set_title('Referencia')
        axes[2].imshow(pr[:128,:128],cmap=cm,vmin=0,vmax=1,interpolation='nearest');axes[2].set_title('Predicción')
        axes[3].imshow(err[:128,:128],cmap=ListedColormap(['#e9eef3','#d43e38','#f4a340','#247c50']),vmin=0,vmax=3,interpolation='nearest');axes[3].set_title('Diagnóstico')
        for a in axes:a.set_xlabel('Columna');a.set_ylabel('Fila');a.set_xticks([0,64,127]);a.set_yticks([0,64,127])
        fig.suptitle(f"Test {i}: {entry['image'].split('/')[-1]} | TP {row.tp} · FP {row.fp} · FN {row.fn}")
        fig.legend(handles=[Patch(color='#b63c78',label='Clase deslizamiento'),Patch(color='#d43e38',label='Falso negativo'),Patch(color='#f4a340',label='Falso positivo'),Patch(color='#247c50',label='Verdadero positivo')],loc='lower center',ncol=4,frameon=False)
        fig.tight_layout(rect=(0,.17,1,.94));fig.savefig(out/f'caso_{typ}.png',dpi=180);plt.close(fig)
    exp.save(out/'seleccion_casos.json',chosen)
    print(comparison.to_string(index=False))


if __name__=='__main__':main()
