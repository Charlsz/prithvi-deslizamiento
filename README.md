# Grupo 2 — Detección de deslizamientos con Prithvi

Trabajo del curso de Inteligencia Artificial (profesor Eduardo Enrique Zurek Varela).  
Segmentación binaria de deslizamientos con Landslide4Sense, encoder Prithvi congelado y Random Forest por parches.

## Entregables del grupo (sin video)

| Entregable | Archivo |
| --- | --- |
| Notebook Jupyter | `Grupo_2_Trabajo_Deslizamientos.ipynb` |
| Informe | `Grupo_2_Informe.md` |
| Métricas, figuras y procedencia | `entregables/` |
| Script reproducible | `run_experimento.py` |
| Verificación automática | `verificar_entrega.py` |

Pendiente externo: video de 15 minutos basado en diapositivas.

## Integrantes

Bauz Villa, Juan; Cera Moran, Daniel; Galvis Pajaro, Carlos; Jimenez Atencia, Cher; Mercado Martinez, Sebastian; Molina Vega, Dayana; Romero Pemberthy, Juan; Ruiz Espitia, Kevin; Vargas Becerra, Natalia.

## Resultado principal del piloto

Con 4 escenas de entrenamiento y 2 de prueba, el modelo no supera a la línea base. Ambos dan IoU 0 y F1 0 en la clase deslizamiento, porque la etiqueta mayoritaria por parche 16×16 elimina la clase positiva antes de entrenar la cabeza.

## Cómo repetir la ejecución

```bash
python -m pip install "torch" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "timm==0.9.2" "huggingface_hub>=0.25,<1" "rasterio>=1.3,<2" "PyYAML>=6,<7" "einops>=0.7,<1" "h5py>=3,<4" "scikit-learn>=1.3,<2" requests pandas matplotlib
python run_experimento.py
python verificar_entrega.py
```

También se puede abrir el notebook en Google Colab y ejecutar todas las celdas en orden.
