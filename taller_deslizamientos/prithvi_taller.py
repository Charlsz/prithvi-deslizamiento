"""Soporte docente experimental para Prithvi. No incluye pesos ni escenas reales.
Las descargas se realizan en Colab. No validado de extremo a extremo.
"""
from pathlib import Path
import json, re, hashlib, tarfile, shutil, warnings, io, time
import numpy as np
import requests

SOURCES = {
 'inundaciones': 'https://github.com/cloudtostreet/Sen1Floods11',
 'quemadas': 'https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars',
 'cultivos': 'https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification',
 'deslizamientos': 'https://huggingface.co/datasets/sasmit545/Landslide4sense',
}

def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def get(url,path,max_mb=250):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size>0:return path
    temp=path.with_suffix(path.suffix+'.part')
    try:
        with requests.get(url,stream=True,timeout=(30,120)) as r:
            r.raise_for_status(); size=0
            with open(temp,'wb') as f:
                for b in r.iter_content(1024*1024):
                    size+=len(b)
                    if size>max_mb*1024**2:raise RuntimeError('Descarga supera el límite por archivo.')
                    f.write(b)
        temp.replace(path)
    finally:
        if temp.exists():temp.unlink()
    return path

def sample_key(path):
    stem=Path(path).name.lower()
    for suffix in ['.tiff','.tif','.h5','.hdf5']:
        if stem.endswith(suffix):stem=stem[:-len(suffix)]
    for token in ['_merged','.mask','_mask','_label','_s2hand','_labelhand']:
        stem=stem.replace(token,'')
    return stem

def pair_tifs(directory):
    files=sorted(Path(directory).rglob('*.tif'))
    labels={sample_key(f):f for f in files if any(s in f.name.lower() for s in ['mask','label'])}
    return [(f,labels[sample_key(f)]) for f in files
            if not any(s in f.name.lower() for s in ['mask','label']) and sample_key(f) in labels]

class LimitedReader:
    def __init__(self, raw, max_bytes):self.raw=raw; self.maximum=max_bytes; self.total=0
    def read(self,size=-1):
        b=self.raw.read(size); self.total+=len(b)
        if self.total>self.maximum:
            raise RuntimeError('El tar excede el presupuesto de red. No se completó el subconjunto. Aumentar max_stream_mb solo tras revisar tamaño y conexión; no hay garantía de descarga pequeña.')
        return b

def subset_archive(url, root, requested, combined=False, max_stream_mb=512):
    """Extrae pares sin descomprimir rutas del tar. Conserva pocos archivos.
    Leer gzip es secuencial: puede transferir mucho aunque se guarde poco.
    """
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    targets={'train':requested[0],'test':requested[1]} if combined else requested
    kept={s:{'image':set(),'label':set()} for s in targets}
    with requests.get(url,stream=True,timeout=(30,120)) as response:
        response.raise_for_status(); response.raw.decode_content=True
        counter=LimitedReader(response.raw,int(max_stream_mb*1024**2))
        with tarfile.open(fileobj=counter,mode='r|gz') as tar:
            for member in tar:
                if not member.isfile() or not member.name.lower().endswith(('.tif','.tiff')):continue
                low=member.name.lower()
                if combined:
                    if re.search(r'(^|[/_])(?:training|train)([/_]|$)',low): split='train'
                    elif re.search(r'(^|[/_])(?:validation|valid|val|test)([/_]|$)',low): split='test'
                    else:continue
                else:split=next(iter(targets))
                target=targets[split]
                if len(pair_tifs(root/split))>=target:continue
                kind='label' if any(t in Path(low).name for t in ['mask','label']) else 'image'
                key=sample_key(low)
                other='image' if kind=='label' else 'label'
                # Guardado acotado; si se listan antes imágenes que máscaras,
                # conservar hasta 4*n para permitir el emparejamiento.
                if key not in kept[split][other] and len(kept[split][kind])>=4*target:continue
                dest=root/split/Path(member.name).name; dest.parent.mkdir(exist_ok=True)
                stream=tar.extractfile(member)
                with open(dest,'wb') as out:shutil.copyfileobj(stream,out)
                kept[split][kind].add(key)
                if all(len(pair_tifs(root/s))>=n for s,n in targets.items()):break
        print('Bytes comprimidos leídos:',counter.total)
    result={s:pair_tifs(root/s)[:n] for s,n in targets.items()}
    for s,n in targets.items():
        if len(result[s])<n:
            raise RuntimeError(f'Solo se emparejaron {len(result[s])}/{n} en {s}. Revisar nombres/estructura del tar; no se sustituyen datos.')
    return result

def download_flood(root,ntrain,ntest):
    import pandas as pd
    root=Path(root); base='https://storage.googleapis.com/sen1floods11/v1.1/'
    result={}
    for split,tag,n in [('train','train',ntrain),('test','test',ntest)]:
        csvname=f'flood_{tag}_data.csv'
        alternatives=[base+'splits/flood_handlabeled/'+csvname,
          'https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/splits/flood_handlabeled/'+csvname]
        content=None
        for url in alternatives:
            rr=requests.get(url,timeout=60)
            if rr.status_code==200:content=rr.text;break
        if content is None:raise RuntimeError('No se pudo recuperar la partición oficial Sen1Floods11.')
        (root).mkdir(parents=True,exist_ok=True); (root/csvname).write_text(content)
        ids=[]
        for line in content.splitlines():
            hit=re.search(r'([A-Za-z]+_\d+)_S[12]Hand\.tif',line)
            if hit and hit.group(1) not in ids:ids.append(hit.group(1))
        if len(ids)<n:raise RuntimeError('La estructura del CSV cambió: revisar antes de continuar.')
        rows=[]
        for sample in ids[:n]:
            image=get(base+'data/flood_events/HandLabeled/S2Hand/'+sample+'_S2Hand.tif',root/split/(sample+'_S2Hand.tif'))
            label=get(base+'data/flood_events/HandLabeled/LabelHand/'+sample+'_LabelHand.tif',root/split/(sample+'_LabelHand.tif'))
            rows.append((image,label))
        result[split]=rows
    return result

def download_landslide(root,ntrain,ntest):
    from huggingface_hub import HfApi,hf_hub_download
    repo='sasmit545/Landslide4sense'; api=HfApi()
    rev=api.dataset_info(repo).sha
    names=api.list_repo_files(repo,repo_type='dataset',revision=rev)
    result={}
    for split,n in [('train',ntrain),('test',ntest)]:
        images=sorted([f for f in names if f.startswith('images/'+split+'/') and f.endswith('.h5')])
        masks={re.findall(r'\d+',Path(f).stem)[-1]:f for f in names
               if f.startswith('annotations/'+split+'/') and f.endswith('.h5')}
        rows=[]
        for im in images:
            idx=re.findall(r'\d+',Path(im).stem)[-1]
            if idx not in masks:continue
            a=hf_hub_download(repo,filename=im,repo_type='dataset',revision=rev,local_dir=str(root))
            b=hf_hub_download(repo,filename=masks[idx],repo_type='dataset',revision=rev,local_dir=str(root))
            rows.append((Path(a),Path(b)))
            if len(rows)==n:break
        if len(rows)<n:raise RuntimeError('Estructura HDF5 diferente de la documentada; revisar el inventario.')
        result[split]=rows
    save_json(Path(root)/'revision_datos.json',{'repo':repo,'revision':rev})
    return result

def download_data(task,root,ntrain=4,ntest=2,max_stream_mb=512):
    from huggingface_hub import HfApi,hf_hub_url
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if task=='inundaciones': result=download_flood(root,ntrain,ntest)
    elif task=='deslizamientos':result=download_landslide(root,ntrain,ntest)
    else:
        repo={'quemadas':'ibm-nasa-geospatial/hls_burn_scars',
              'cultivos':'ibm-nasa-geospatial/multi-temporal-crop-classification'}[task]
        api=HfApi();revision=api.dataset_info(repo).sha
        files=api.list_repo_files(repo,repo_type='dataset',revision=revision)
        print('Inventario del dataset:',files)
        if task=='quemadas':
            archive=next((f for f in files if f.endswith('hls_burn_scars.tar.gz')),None)
            if archive is None:raise RuntimeError('No se encontró el tar documentado de Burn Scars.')
            result=subset_archive(hf_hub_url(repo,archive,repo_type='dataset',revision=revision),root,
                                  (ntrain,ntest),combined=True,max_stream_mb=max_stream_mb)
        else:
            result={}
            for split,name,n in [('train','training_chips.tgz',ntrain),('test','validation_chips.tgz',ntest)]:
                if name not in files:raise RuntimeError(f'Falta {name}; revisar inventario.')
                result.update(subset_archive(hf_hub_url(repo,name,repo_type='dataset',revision=revision),root,
                         {split:n},combined=False,max_stream_mb=max_stream_mb))
        save_json(root/'revision_datos.json',{'repo':repo,'revision':revision})
    manifest=[]
    for split,pairs in result.items():
        for im,lab in pairs:
            manifest.append({'split':split,'image':str(im),'label':str(lab),
                             'image_sha256':sha256(im),'label_sha256':sha256(lab)})
    save_json(root/'manifest.json',{'task':task,'source':SOURCES[task],
          'note':'Subconjunto por orden, no por desempeño; train/test según proveedor. En cultivos, validation se usa como evaluación sin ajustar decisiones.',
          'samples':manifest})
    return result

def read_sample(task, image_path, label_path, crop_layout='CT',side=224):
    """Retorna imagen C,T,H,W y etiqueta H,W. No remuestrea etiquetas.
    Orden óptico asumido/documentado en el notebook: azul,verde,rojo,B8A,SWIR1,SWIR2.
    """
    import rasterio,h5py
    if task=='deslizamientos':
        with h5py.File(image_path,'r') as f:
            keys=list(f.keys());key='img' if 'img' in f else 'image' if 'image' in f else None
            if key is None:raise RuntimeError(f'Clave HDF5 no reconocida: {keys}')
            im=np.asarray(f[key],dtype=np.float32)
        if im.shape[-1]==14:im=im.transpose(2,0,1)
        if im.shape[0]!=14:raise ValueError('Se esperaban 14 canales Landslide4Sense.')
        im=im[[1,2,3,8,10,11]][:,None] # B02 B03 B04 B8A B11 B12; excluye relieve
        with h5py.File(label_path,'r') as f:
            if 'mask' not in f:raise RuntimeError('No existe la clave mask en la referencia.')
            label=np.asarray(f['mask']).squeeze().astype(np.int64)
    else:
        with rasterio.open(image_path) as f:
            arr=f.read(masked=True).astype(np.float32).filled(np.nan)
            ref_crs,ref_transform=f.crs,f.transform
        with rasterio.open(label_path) as f:
            label=f.read(1,masked=True).astype(np.int64).filled(-1)
            if f.crs!=ref_crs or f.transform!=ref_transform:raise ValueError('Imagen y etiqueta geográficamente desalineadas.')
        if task=='inundaciones':
            if arr.shape[0]!=13:raise ValueError('Se esperan las 13 bandas S2Hand en orden B01..B12, incluyendo B8A.')
            im=arr[[1,2,3,8,11,12]][:,None]
        elif task=='quemadas':
            if arr.shape[0]!=6:raise ValueError('Se esperan 6 bandas HLS.')
            im=arr[:,None]
        else:
            if arr.shape[0]!=18:raise ValueError('Se esperan 18 canales (6 bandas × 3 fechas).')
            if crop_layout=='CT':im=arr.reshape(6,3,*arr.shape[1:])
            elif crop_layout=='TC':im=arr.reshape(3,6,*arr.shape[1:]).transpose(1,0,2,3)
            else:raise ValueError('crop_layout debe ser CT o TC; confirmar la documentación de la versión.')
            label=np.where(label==0,-1,label) # código 0 = nodata; no desplaza clases 1..13
    if im.shape[-2:]!=label.shape:raise ValueError('Tamaño de etiqueta incompatible.')
    valid=np.isfinite(im).all(axis=(0,1)) & (im!=-9999).all(axis=(0,1))
    label=np.where(valid,label,-1)
    h,w=label.shape; r=max(0,(h-side)//2);c=max(0,(w-side)//2)
    im=im[:,:,r:r+side,c:c+side];label=label[r:r+side,c:c+side]
    ph=side-im.shape[-2];pw=side-im.shape[-1]
    # Bordes de imágenes menores se repiten para contexto; la etiqueta de relleno es ignorada.
    if ph or pw:
        im=np.pad(im,((0,0),(0,0),(0,ph),(0,pw)),mode='edge')
        label=np.pad(label,((0,ph),(0,pw)),constant_values=-1)
    return im.astype('float32'),label

def train_normalization(samples):
    # Estadísticas de entrenamiento exclusivamente, por banda y todas sus fechas.
    vals=[np.where(y[None,None]>=0,x,np.nan).reshape(6,-1) for x,y in samples]
    joined=np.concatenate(vals,axis=1)
    mean=np.nanmean(joined,axis=1); std=np.nanstd(joined,axis=1)
    if not np.isfinite(mean).all():raise ValueError('No hay datos válidos para normalizar.')
    return mean,np.maximum(std,1e-6)

def normalize(x,mean,std):
    x=(x-mean[:,None,None,None])/std[:,None,None,None]
    return np.nan_to_num(x,nan=0,posinf=0,neginf=0).astype('float32')

def load_prithvi(root,num_frames=1):
    import torch,yaml,sys,importlib.util
    from huggingface_hub import HfApi,hf_hub_download
    repo='ibm-nasa-geospatial/Prithvi-EO-1.0-100M'
    revision='7b231fe5b0684e2ddd40c967fbc02e225ef77dc6'
    names=HfApi().list_repo_files(repo,revision=revision)
    def choose(options):
        for f in options:
            if f in names:return f
        raise RuntimeError('No está el archivo esperado en la revisión fijada: '+str(options))
    py=choose(['Prithvi.py','prithvi.py','prithvi_mae.py'])
    cf=choose(['Prithvi_100M_config.yaml','config.yaml'])
    wt=choose(['Prithvi_100M.pt'])
    def dl(f):return Path(hf_hub_download(repo,filename=f,revision=revision,local_dir=str(root)))
    py=dl(py); cf=dl(cf); wt=dl(wt)
    # Ejecuta Python remoto del repositorio oficial, fijado a revisión. Revisar antes de distribuir.
    spec=importlib.util.spec_from_file_location('prithvi_official',py)
    mod=importlib.util.module_from_spec(spec);sys.path.insert(0,str(py.parent));spec.loader.exec_module(mod)
    klass=next((getattr(mod,n) for n in ['MaskedAutoencoderViT','PrithviMAE'] if hasattr(mod,n)),None)
    if klass is None:raise RuntimeError('Clase MAE no encontrada.')
    cfg=yaml.safe_load(cf.read_text())
    if isinstance(cfg, dict) and 'model_args' in cfg:
        args=dict(cfg['model_args'])
    else:
        allowed={'img_size','patch_size','num_frames','tubelet_size','in_chans','embed_dim','depth','num_heads','decoder_embed_dim','decoder_depth','decoder_num_heads','mlp_ratio','norm_pix_loss'}
        args={k:v for k,v in cfg.items() if k in allowed}
        if 'bands' in cfg and 'in_chans' not in args:
            args['in_chans']=len(cfg['bands'])
    args['num_frames']=num_frames
    if 'in_chans' not in args:
        args['in_chans']=6
    model=klass(**args)
    state=torch.load(wt,map_location='cpu',weights_only=True)
    if 'state_dict' in state:state=state['state_dict']
    if 'model' in state and isinstance(state['model'],dict):state=state['model']
    state=dict(state)
    for key in ['pos_embed','decoder_pos_embed']:state.pop(key,None)
    status=model.load_state_dict(state,strict=False)
    if set(status.missing_keys)-{'pos_embed','decoder_pos_embed'} or status.unexpected_keys:
        raise RuntimeError('Checkpoint incompatible: '+str(status))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=model.to(device).eval()
    for param in model.parameters():param.requires_grad=False
    return model,device,{'repo':repo,'revision':revision,'checkpoint':str(wt),'device':device,'num_frames':num_frames}

def features(model,device,image):
    import torch
    x=torch.from_numpy(image.copy())[None].to(device)
    torch.manual_seed(0)
    with torch.inference_mode():
        latent,mask,restore=model.forward_encoder(x,mask_ratio=0.)
        tokens=latent[:,1:,:]
        # Aunque mask_ratio=0, MAE puede permutar los parches. Recuperar orden espacial.
        tokens=torch.gather(tokens,1,restore.unsqueeze(-1).expand(-1,-1,tokens.shape[-1]))
    T=image.shape[1];D=tokens.shape[-1]
    if tokens.shape[1]!=T*14*14:raise RuntimeError('Longitud de tokens distinta de la esperada (224, parches16).')
    return tokens.reshape(T,14,14,D).mean(dim=0).cpu().numpy()

def patch_labels(label,size=16,min_fraction=.5):
    result=np.full((14,14),-1,dtype=np.int64)
    for r in range(14):
        for c in range(14):
            block=label[r*size:(r+1)*size,c*size:(c+1)*size]
            v=block[block>=0]
            if v.size>=min_fraction*size*size:
                classes,counts=np.unique(v,return_counts=True)
                result[r,c]=classes[np.argmax(counts)]
    return result

def confusion(y,pred,classes):
    y=np.asarray(y); pred=np.asarray(pred)
    if y.shape!=pred.shape:raise ValueError('Formas distintas.')
    valid=y>=0; a=y[valid];b=pred[valid]
    if not valid.any():raise ValueError('No hay píxeles válidos.')
    if not np.isin(a,classes).all() or not np.isin(b,classes).all():raise ValueError('Clase fuera de la leyenda.')
    mat=np.zeros((len(classes),len(classes)),dtype=np.int64)
    for i,c in enumerate(classes):
        for j,d in enumerate(classes):mat[i,j]=np.count_nonzero((a==c)&(b==d))
    return mat

def metrics(mat,classes):
    rows=[]
    for i,cl in enumerate(classes):
        tp=int(mat[i,i]); fp=int(mat[:,i].sum()-tp); fn=int(mat[i,:].sum()-tp)
        support=int(mat[i,:].sum());union=tp+fp+fn
        iou=tp/union if union else np.nan
        f1=2*tp/(2*tp+fp+fn) if union else np.nan
        precision=tp/(tp+fp) if tp+fp else (0. if support else np.nan)
        recall=tp/(tp+fn) if support else np.nan
        rows.append({'class':int(cl),'support':support,'iou':iou,'f1':f1,'precision':precision,'recall':recall})
    return rows

def smoke_test():
    y=np.array([[0,0],[1,-1]]);p=np.array([[0,1],[1,0]])
    mat=confusion(y,p,[0,1]);assert np.array_equal(mat,[[1,1],[0,1]])
    row=metrics(mat,[0,1])[1]
    assert np.isclose(row['iou'],.5) and np.isclose(row['f1'],2/3)
    # nodata debe excluirse; nunca se cuenta como fondo.
    assert mat.sum()==3
    return 'Tests de métricas superados; NO prueban descarga ni inferencia.'
