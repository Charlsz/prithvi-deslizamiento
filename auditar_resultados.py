"""Diagnóstico trazable del piloto docente; no ajusta el modelo ni usa test para selección."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def auditar(root, out):
    from prithvi_taller import read_sample, patch_labels, sha256
    root, out = Path(root), Path(out)
    manifest = json.loads((out / 'manifest_datos.json').read_text(encoding='utf-8'))
    records, patches, cases = [], [], []
    split_hashes = {'train': set(), 'test': set()}
    for entry in manifest['samples']:
        split = entry['split']
        image = root / 'datos' / 'images' / split / Path(entry['image']).name
        mask = root / 'datos' / 'annotations' / split / Path(entry['label']).name
        assert sha256(image) == entry['image_sha256']
        assert sha256(mask) == entry['label_sha256']
        split_hashes[split].add(entry['image_sha256'])
        x, y = read_sample('deslizamientos', image, mask)
        valid = y >= 0
        yy, xx = np.where(y == 1)
        records.append(dict(split=split, image=image.name, mask=mask.name,
            valid=int(valid.sum()), positive=int((y == 1).sum()),
            ignored=int((~valid).sum()), positive_fraction=float((y == 1).sum()/valid.sum()),
            positive_bbox=str([int(yy.min()), int(xx.min()), int(yy.max()), int(xx.max())]) if len(yy) else 'absent'))
        pl = patch_labels(y)
        for r in range(14):
            for c in range(14):
                block = y[r*16:(r+1)*16, c*16:(c+1)*16]
                n = int((block >= 0).sum())
                if n:
                    patches.append(dict(split=split, image=image.name, row=r, column=c,
                        valid=n, positive=int((block == 1).sum()),
                        positive_fraction=float((block == 1).sum()/n), target=int(pl[r,c])))
        if split != 'test':
            continue
        i = sum(z['split'] == 'test' for z in records)-1
        saved = np.load(out / f'escena_{i}.npz')
        assert np.array_equal(y, saved['label'])
        pred = saved['prediction']
        error = np.zeros(y.shape, dtype=int)
        error[(y == 1) & (pred == 0)] = 1
        error[(y == 0) & (pred == 1)] = 2
        error[(y == 1) & (pred == 1)] = 3
        for typ, condition in [('TN', (y == 0) & (pred == 0)), ('FN', (y == 1) & (pred == 0)),
                               ('FP', (y == 0) & (pred == 1)), ('TP', (y == 1) & (pred == 1))]:
            rows, cols = np.where(condition)
            cases.append(dict(scene=i, image=image.name, case=typ, pixels=len(rows),
                example_row=int(rows[len(rows)//2]) if len(rows) else -1,
                example_column=int(cols[len(rows)//2]) if len(rows) else -1))
        rgb = x[[2,1,0],0].transpose(1,2,0).copy()[:128,:128]
        for channel in range(3):
            lo, hi = np.percentile(rgb[:,:,channel], [2,98])
            rgb[:,:,channel] = np.clip((rgb[:,:,channel]-lo)/max(hi-lo,1e-6),0,1)
        fig, ax = plt.subplots(1,4,figsize=(13,3.8))
        ax[0].imshow(rgb); ax[0].set_title('RGB de visualización')
        cm = ListedColormap(['#e9eef3','#b63c78'])
        ax[1].imshow(y[:128,:128], cmap=cm, vmin=0,vmax=1); ax[1].set_title('Referencia 0 / 1')
        ax[2].imshow(pred[:128,:128], cmap=cm,vmin=0,vmax=1); ax[2].set_title('Predicción 0 / 1')
        ax[3].imshow(error[:128,:128], cmap=ListedColormap(['#e9eef3','#d43e38','#f4a340','#247c50']),vmin=0,vmax=3)
        ax[3].set_title('Mapa de errores')
        for a in ax:
            a.set_xlabel('Columna (base cero)'); a.set_ylabel('Fila (base cero)')
        fig.suptitle(f'Test {i}: {image.name} | {int((y==1).sum())} píxeles positivos')
        fig.legend(handles=[Patch(color='#d43e38',label='Falso negativo'),Patch(color='#f4a340',label='Falso positivo'),Patch(color='#247c50',label='Verdadero positivo')],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.01))
        fig.tight_layout(rect=(0,.19,1,.94)); fig.savefig(out/f'04_trazabilidad_{i}.png',dpi=170); plt.close(fig)
        # Acercamiento con idéntica extensión para distinguir objetos pequeños.
        if len(yy):
            r0,r1=max(0,int(yy.min())-8),min(128,int(yy.max())+9)
            c0,c1=max(0,int(xx.min())-8),min(128,int(xx.max())+9)
            fig, ax = plt.subplots(1,3,figsize=(9,3.5))
            extent=(c0-.5,c1-.5,r1-.5,r0-.5)
            for a, arr, title in zip(ax,[rgb[r0:r1,c0:c1],y[r0:r1,c0:c1],pred[r0:r1,c0:c1]],['RGB','Referencia','Predicción']):
                if arr.ndim==3:a.imshow(arr,extent=extent)
                else:a.imshow(arr,cmap=cm,vmin=0,vmax=1,extent=extent,interpolation='nearest')
                a.set_title(title);a.set_xlabel('Columna');a.set_ylabel('Fila')
            fig.suptitle(f'Acercamiento de falsos negativos | {image.name}')
            fig.tight_layout();fig.savefig(out/f'05_detalle_FN_{i}.png',dpi=170);plt.close(fig)
    assert not split_hashes['train'] & split_hashes['test'], 'Imágenes duplicadas entre train y test'
    pd.DataFrame(records).to_csv(out/'inventario_muestras.csv',index=False)
    df=pd.DataFrame(patches);df.to_csv(out/'diagnostico_parches.csv',index=False)
    pd.DataFrame(cases).to_csv(out/'casos_errores.csv',index=False)
    train=df[df.split=='train']
    diagnostic=dict(train_patch_targets={str(int(k)):int(v) for k,v in train[train.target>=0].target.value_counts().items()},
        max_train_positive_fraction=float(train.positive_fraction.max()),
        train_test_duplicate_images=False,
        metadata='HDF5 sin atributos de fecha, CRS, geotransformación ni identificador de evento; no se acredita independencia por evento.',
        channel_contract='Selección docente [1,2,3,8,10,11]. La descripción pública de 12 bandas no identifica inequívocamente B8A; pendiente confirmación docente.',
        teacher_confirmation='Pendiente de evidencia',
        interpretation='La falta de clase positiva en la cabeza impide evaluar la capacidad discriminativa de Prithvi para deslizamientos.')
    (out/'diagnostico.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(diagnostic,ensure_ascii=False,indent=2))
    return diagnostic


if __name__ == '__main__':
    import sys
    root=Path(__file__).resolve().parent/'taller_deslizamientos'
    sys.path.insert(0,str(root))
    auditar(root,root/'resultados')
