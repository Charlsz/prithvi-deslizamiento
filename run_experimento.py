"""Ejecuta el experimento completo de deslizamientos de forma local."""
from __future__ import annotations

import json
import shutil
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

ROOT_REPO = Path(__file__).resolve().parent
WORK = ROOT_REPO / "taller_deslizamientos"
OUT = WORK / "resultados"
MODULE_PATH = WORK / "prithvi_taller.py"
NB_PATH = ROOT_REPO / "Grupo_2_Trabajo_Deslizamientos.ipynb"

TASK = "deslizamientos"
CLASSES = [0, 1]
NTRAIN = 4
NTEST = 2
MAX_STREAM_MB = 512
CROP_LAYOUT = "CT"


def extract_module_text() -> str:
    data = json.loads(NB_PATH.read_text(encoding="utf-8"))
    for cell in data["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "MODULE_TEXT" not in src or "def smoke_test" not in src:
            continue
        markers = [
            "MODULE_FILE =",
            "Path('/content/prithvi_taller.py')",
            'Path("/content/prithvi_taller.py")',
        ]
        cut = None
        for marker in markers:
            if marker in src:
                cut = src.split(marker)[0]
                break
        if cut is None:
            raise RuntimeError("No se encontró la escritura del módulo en el notebook.")
        local: dict = {}
        exec(compile(cut, "<module_cell>", "exec"), {}, local)
        return local["MODULE_TEXT"]
    raise RuntimeError("No se encontró MODULE_TEXT en el notebook.")


def display_rgb(x, label=None):
    """RGB en ventana válida; oculta relleno edge usado para completar 224."""
    a = x[[2, 1, 0], 0].transpose(1, 2, 0).copy()
    for c in range(3):
        v = a[:, :, c]
        finite = np.isfinite(v)
        if label is not None:
            finite = finite & (label >= 0)
        if not finite.any():
            continue
        lo, hi = np.percentile(v[finite], [2, 98])
        a[:, :, c] = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    a = np.nan_to_num(a)
    if label is not None:
        mask = label < 0
        a[mask] = 0.15
    return a


def valid_bbox(label):
    ys, xs = np.where(label >= 0)
    if len(ys) == 0:
        return slice(None), slice(None)
    return slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (WORK / "datos").mkdir(parents=True, exist_ok=True)
    (WORK / "modelo").mkdir(parents=True, exist_ok=True)

    module_text = extract_module_text()
    # Parche necesario solo si el notebook aún trae el YAML anidado original.
    old_cfg = "cfg=yaml.safe_load(cf.read_text());args=dict(cfg['model_args']);args['num_frames']=num_frames\n    model=klass(**args)"
    new_cfg = (
        "cfg=yaml.safe_load(cf.read_text())\n"
        "    if isinstance(cfg, dict) and 'model_args' in cfg:\n"
        "        args=dict(cfg['model_args'])\n"
        "    else:\n"
        "        allowed={'img_size','patch_size','num_frames','tubelet_size','in_chans','embed_dim','depth','num_heads','decoder_embed_dim','decoder_depth','decoder_num_heads','mlp_ratio','norm_pix_loss'}\n"
        "        args={k:v for k,v in cfg.items() if k in allowed}\n"
        "        if 'bands' in cfg and 'in_chans' not in args:\n"
        "            args['in_chans']=len(cfg['bands'])\n"
        "    args['num_frames']=num_frames\n"
        "    if 'in_chans' not in args:\n"
        "        args['in_chans']=6\n"
        "    model=klass(**args)"
    )
    if old_cfg in module_text:
        module_text = module_text.replace(old_cfg, new_cfg)
    elif "in_chans" not in module_text:
        raise RuntimeError("El módulo no tiene ni el bloque original ni el parche de in_chans.")
    MODULE_PATH.write_text(module_text, encoding="utf-8")
    if "prithvi_taller" in sys.modules:
        del sys.modules["prithvi_taller"]
    sys.path.insert(0, str(WORK))
    from prithvi_taller import (
        confusion,
        download_data,
        features,
        load_prithvi,
        metrics,
        normalize,
        patch_labels,
        read_sample,
        save_json,
        smoke_test,
        train_normalization,
    )

    print(smoke_test())
    print("Tarea:", TASK, "| dispositivo:", "GPU" if torch.cuda.is_available() else "CPU")

    pairs = download_data(TASK, WORK / "datos", ntrain=NTRAIN, ntest=NTEST, max_stream_mb=MAX_STREAM_MB)
    for split, rows in pairs.items():
        print(split, [(im.name, lab.name) for im, lab in rows])

    samples = {
        split: [read_sample(TASK, im, lab, crop_layout=CROP_LAYOUT) for im, lab in rows]
        for split, rows in pairs.items()
    }
    for split, rows in samples.items():
        print(split, [(x.shape, y.shape, np.unique(y).tolist()) for x, y in rows])
        for _, y in rows:
            if not np.isin(y[y >= 0], CLASSES).all():
                raise ValueError("Etiqueta fuera de leyenda; revisar la versión.")

    x0, y0 = samples["train"][0]
    ry, rx = valid_bbox(y0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(display_rgb(x0, y0)[ry, rx])
    ax[0].set_title("RGB de entrenamiento (zona válida)")
    ax[1].imshow(y0[ry, rx], vmin=min(CLASSES), vmax=max(CLASSES), cmap="tab20")
    ax[1].set_title("Referencia; ignorados ocultos")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "01_exploracion.png", dpi=150)
    plt.close(fig)

    mean, std = train_normalization(samples["train"])
    prepared = {
        split: [(normalize(x, mean, std), y) for x, y in rows]
        for split, rows in samples.items()
    }
    save_json(OUT / "normalizacion.json", {
        "method": "train_zscore_by_band",
        "mean": mean.tolist(),
        "std": std.tolist(),
    })
    print("Media train:", mean, "| STD train:", std)

    T = prepared["train"][0][0].shape[1]
    model, device, model_info = load_prithvi(WORK / "modelo", num_frames=T)
    provenance = {
        "task": TASK,
        "model": model_info,
        "ntrain": NTRAIN,
        "ntest": NTEST,
        "classes": CLASSES,
        "crop_layout": CROP_LAYOUT,
        "crop": "center224_pad_ignore",
        "normalization": "train_zscore_by_band",
        "head": "RandomForest, patch16 majority, frozen encoder",
        "packages": {
            p: version(p)
            for p in ["torch", "torchvision", "timm", "numpy", "scikit-learn", "rasterio"]
        },
        "execution": {
            "environment": "local_windows",
            "device": device,
            "workspace": str(WORK),
        },
    }
    save_json(OUT / "procedencia.json", provenance)
    print(model_info)

    train_features = []
    train_targets = []
    for i, (x, y) in enumerate(prepared["train"]):
        f = features(model, device, x)
        label = patch_labels(y)
        valid = label >= 0
        train_features.append(f[valid])
        train_targets.append(label[valid])
        print("Entrenamiento escena", i, "parches válidos:", int(valid.sum()))

    X = np.concatenate(train_features)
    Y = np.concatenate(train_targets)
    if len(Y) == 0:
        raise ValueError("No hay parches de entrenamiento válidos.")
    print("Distribución de etiquetas por parche:", np.unique(Y, return_counts=True))
    missing = sorted(set(CLASSES) - set(np.unique(Y)))
    if missing:
        print("LIMITACIÓN: clases ausentes del entrenamiento:", missing)

    head = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=2,
    )
    assert np.isfinite(X).all(), "Embeddings contienen NaN o infinito"
    head.fit(X, Y)

    train_labels = np.concatenate([y[y >= 0] for _, y in samples["train"]])
    values, counts = np.unique(train_labels, return_counts=True)
    majority = int(values[np.argmax(counts)])
    print("Línea base constante:", majority)

    totals = {
        "prithvi_rf": np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64),
        "baseline": np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64),
    }
    scene_rows = []
    for i, ((x, y), (raw, _)) in enumerate(zip(prepared["test"], samples["test"])):
        f = features(model, device, x)
        grid = head.predict(f.reshape(-1, f.shape[-1])).reshape(f.shape[0], f.shape[1])
        pred = np.repeat(np.repeat(grid, 16, axis=0), 16, axis=1)
        baseline = np.full(y.shape, majority, dtype=np.int64)
        for name, p in [("prithvi_rf", pred), ("baseline", baseline)]:
            mat = confusion(y, p, CLASSES)
            totals[name] += mat
            for row in metrics(mat, CLASSES):
                scene_rows.append({"scene": i, "method": name, **row})
        np.savez_compressed(OUT / f"escena_{i}.npz", label=y, prediction=pred, baseline=baseline)

        ry, rx = valid_bbox(y)
        err = (pred != y).astype(float)
        fig, ax = plt.subplots(1, 4, figsize=(15, 4))
        ax[0].imshow(display_rgb(raw, y)[ry, rx])
        ax[0].set_title("RGB fecha 1 (zona válida)")
        ax[1].imshow(y[ry, rx], vmin=min(CLASSES), vmax=max(CLASSES), cmap="tab20")
        ax[1].set_title("Referencia")
        ax[2].imshow(pred[ry, rx], vmin=min(CLASSES), vmax=max(CLASSES), cmap="tab20")
        ax[2].set_title("Prithvi + RF (parche16)")
        ax[3].imshow(err[ry, rx], vmin=0, vmax=1, cmap="Reds")
        ax[3].set_title("Errores en rojo")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / f"02_resultado_{i}.png", dpi=150)
        plt.close(fig)

        # Resumen cuantitativo de errores por escena
        valid = y >= 0
        tp = int(((pred == 1) & (y == 1) & valid).sum())
        fp = int(((pred == 1) & (y == 0) & valid).sum())
        fn = int(((pred == 0) & (y == 1) & valid).sum())
        tn = int(((pred == 0) & (y == 0) & valid).sum())
        print(f"Escena {i}: TP={tp} FP={fp} FN={fn} TN={tn}")

    rows = []
    for name, mat in totals.items():
        for row in metrics(mat, CLASSES):
            rows.append({"method": name, **row})
        pd.DataFrame(mat, index=CLASSES, columns=CLASSES).to_csv(OUT / f"confusion_{name}.csv")
    byclass = pd.DataFrame(rows)
    byclass.to_csv(OUT / "metricas_por_clase.csv", index=False)
    pd.DataFrame(scene_rows).to_csv(OUT / "metricas_por_escena.csv", index=False)

    summary = []
    for name in totals:
        subset = byclass[byclass.method == name]
        pos = subset[subset["class"] == 1].iloc[0]
        summary.append({
            "method": name,
            "f1_positive": pos.f1,
            "iou_positive": pos.iou,
            "precision": pos.precision,
            "recall": pos.recall,
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "metricas.csv", index=False)
    print(summary)

    scene_df = pd.DataFrame(scene_rows)
    positive = scene_df[scene_df["class"] == 1].copy()
    metric_names = ["iou", "f1", "precision", "recall"]
    mean_scene = positive.groupby("method")[metric_names].mean().reset_index()
    counts_defined = (
        positive.groupby("method")[metric_names]
        .count()
        .add_prefix("escenas_definidas_")
        .reset_index()
    )
    mean_scene = mean_scene.merge(counts_defined, on="method")
    mean_scene.to_csv(OUT / "metricas_media_por_escena.csv", index=False)
    print("Media por escena:")
    print(mean_scene)

    train_distribution = pd.DataFrame({"clase": values, "pixeles": counts})
    train_distribution["fraccion"] = train_distribution.pixeles / train_distribution.pixeles.sum()
    train_distribution.to_csv(OUT / "distribucion_train.csv", index=False)

    plot_df = mean_scene.set_index("method")[metric_names]
    ax = plot_df.T.plot.bar(
        figsize=(9, 4),
        rot=0,
        ylim=(0, 1.05),
        title="Comparación por clase positiva (media por escena)",
    )
    ax.set_ylabel("Valor de la métrica")
    ax.legend(title="Método")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2)
    ax.text(
        0.02,
        0.92,
        "Nota: barras en 0 indican que la clase deslizamiento no fue predicha.",
        transform=ax.transAxes,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(OUT / "03_comparacion_metricas.png", dpi=160)
    plt.close()

    shutil.copy2(WORK / "datos" / "manifest.json", OUT / "manifest_datos.json")
    evidence = shutil.make_archive(str(WORK / "evidencias"), "zip", OUT)
    datazip = shutil.make_archive(str(WORK / "subconjunto_real"), "zip", WORK / "datos")
    print("Evidencias:", evidence)
    print("Datos reales:", datazip)

    required = [
        "metricas.csv",
        "metricas_por_clase.csv",
        "metricas_por_escena.csv",
        "metricas_media_por_escena.csv",
        "distribucion_train.csv",
        "01_exploracion.png",
        "02_resultado_0.png",
        "02_resultado_1.png",
        "03_comparacion_metricas.png",
        "procedencia.json",
        "normalizacion.json",
        "manifest_datos.json",
        "confusion_prithvi_rf.csv",
        "confusion_baseline.csv",
    ]
    missing_files = [f for f in required if not (OUT / f).exists()]
    report = {
        "ok": len(missing_files) == 0,
        "missing": missing_files,
        "summary": summary.to_dict(orient="records"),
        "mean_scene": mean_scene.to_dict(orient="records"),
        "majority_baseline": majority,
        "train_distribution": train_distribution.to_dict(orient="records"),
        "device": device,
    }
    save_json(OUT / "verificacion_ejecucion.json", report)
    print("VERIFICACION:", report["ok"], "faltan:", missing_files)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
