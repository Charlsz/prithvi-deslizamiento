"""Verifica dos veces que los entregables del Grupo 2 estén completos (sin video)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = {
    "notebook": ROOT / "Grupo_2_Trabajo_Deslizamientos.ipynb",
    "informe": ROOT / "Grupo_2_Informe.md",
    "script": ROOT / "run_experimento.py",
}
REQUIRED_ENTREGABLES = [
    "metricas.csv",
    "metricas_por_clase.csv",
    "metricas_por_escena.csv",
    "metricas_media_por_escena.csv",
    "distribucion_train.csv",
    "confusion_prithvi_rf.csv",
    "confusion_baseline.csv",
    "01_exploracion.png",
    "02_resultado_0.png",
    "02_resultado_1.png",
    "03_comparacion_metricas.png",
    "procedencia.json",
    "normalizacion.json",
    "manifest_datos.json",
]
INTEGRANTES = [
    "bauz villa",
    "cera moran",
    "galvis pajaro",
    "jimenez atencia",
    "mercado martinez",
    "molina vega",
    "romero pemberthy",
    "ruiz espitia",
    "vargas becerra",
]


def check_once(tag: str) -> bool:
    ok = True
    print(f"=== Verificación {tag} ===")
    for name, path in REQUIRED.items():
        exists = path.exists() and path.stat().st_size > 0
        print(f"{name}: {'OK' if exists else 'FALTA'} ({path.name})")
        ok &= exists

    entregables = ROOT / "entregables"
    for fname in REQUIRED_ENTREGABLES:
        path = entregables / fname
        exists = path.exists() and path.stat().st_size > 0
        print(f"entregable {fname}: {'OK' if exists else 'FALTA'}")
        ok &= exists

    metricas = list(csv.DictReader((entregables / "metricas.csv").open(encoding="utf-8")))
    methods = {row["method"] for row in metricas}
    print("métodos en metricas.csv:", sorted(methods))
    ok &= methods == {"prithvi_rf", "baseline"}

    informe = (ROOT / "Grupo_2_Informe.md").read_text(encoding="utf-8").lower()
    nb = json.loads((ROOT / "Grupo_2_Trabajo_Deslizamientos.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(c.get("source", [])) for c in nb["cells"]).lower()
    both = informe + "\n" + text

    for needle in [
        "iou",
        "f1",
        "línea base",
        "colombia",
        "falso negativo",
        "prithvi",
        "consideraciones éticas",
        "arxiv:2412.02732",
        "formato ieee",
    ]:
        present = needle in both
        print(f"contenido '{needle}': {'OK' if present else 'FALTA'}")
        ok &= present

    for nombre in INTEGRANTES:
        present = nombre in both
        print(f"integrante '{nombre}': {'OK' if present else 'FALTA'}")
        ok &= present

    signals = [
        ("parche yaml", "in_chans" in text),
        ("respuesta deteccion", "detectar" in text and "susceptibilidad" in text),
        ("propuesta colombia", "antioquia" in text or "colombia" in text),
        ("conclusiones", "conclusiones" in text),
    ]
    for label, present in signals:
        print(f"notebook {label}: {'OK' if present else 'FALTA'}")
        ok &= present

    print(f"Resultado {tag}: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def main() -> int:
    first = check_once("1")
    second = check_once("2")
    print("Ambas verificaciones OK" if first and second else "Hay pendientes")
    print("Excluido a propósito: video de 15 minutos")
    return 0 if first and second else 1


if __name__ == "__main__":
    raise SystemExit(main())
