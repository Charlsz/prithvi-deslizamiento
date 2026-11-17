# Detección de deslizamientos con Prithvi

Proyecto del Grupo 2 de Inteligencia Artificial, profesor Eduardo Enrique Zurek Varela. Entrega del 23 de noviembre de 2026. Revisión y ejecución del 17 de noviembre de 2026.

## Archivos para revisar y entregar

| Archivo | Contenido |
| --- | --- |
| `Grupo_2_Trabajo_Deslizamientos.ipynb` | Entregable principal solicitado: fundamentos, contribuciones, código ejecutado, resultados, análisis, transferencia, ética y referencias IEEE |
| `Grupo_2_Informe.pdf` | Informe legible con tablas y figuras reales |
| `Grupo_2_Informe.md` | Fuente editable del informe |
| `entregables/` | CSV, figuras, máscaras y predicciones NPZ, procedencia y verificación |
| `REVISION_ENTREGA.md` | Correspondencia con la rúbrica y pendientes que requieren al docente |
| `run_experimento.py` | Ejecuta el notebook completo con un kernel nuevo y guarda las salidas |
| `verificar_entrega.py` | Recalcula métricas con scikit-learn y comprueba la coherencia de la entrega |

El paquete local `Entrega_Grupo_2_sin_video.zip` reúne estos archivos con el código de apoyo. Excluye pesos y escenas crudas; el notebook los descarga desde fuentes con revisión fijada. No se incluye el video, a cargo del grupo. El enunciado exige publicar el notebook en GitHub: preparar archivos locales no equivale a publicar los cambios.

## Resultado y límites

La ampliación de `experimentos/mejora_rf/` añade resultados distintos de cero con 96 escenas de entrenamiento, 24 de validación y 32 de prueba nuevas. El RF mayoritario con umbral validado obtiene F1 medio 0.1944 e IoU media 0.1323. La extensión por píxel con Prithvi y bandas ópticas obtiene F1 medio 0.3068 e IoU media 0.2030. Son resultados moderados; no se afirma desempeño alto ni aprobación docente de una ruta alternativa. El control óptico también se reporta, incluso cuando supera a Prithvi. Consulte `REPRODUCIR_MEJORA.md` y la sección 12 del informe.

```powershell
python verificar_mejora.py
```

El flujo docente conserva cuatro escenas de entrenamiento y dos de prueba. IoU y F1 de deslizamiento son 0 para Prithvi con Random Forest y para la línea base. Los 919 píxeles positivos de entrenamiento se convierten en cero parches positivos: los 256 objetivos de parche son fondo. El máximo de ocupación positiva de un parche es 39.84375 %. No se concluye que Prithvi sea incapaz de detectar deslizamientos con una adaptación distinta.

La agregación principal es media por escena (`metricas_media_por_escena.csv`); `metricas.csv` contiene la agregación global por píxeles. El verificador comprueba ambas de forma independiente. La confirmación docente de canales, escala, normalización y pesos sigue pendiente; la ficha la exige y el profesor aún no ha comunicado una aclaración.

## Reproducir

Entorno comprobado: Python 3.11 en Windows, CPU. Las versiones exactas utilizadas están en `requirements.txt`; los pesos ocupan cientos de MB y requieren conexión. Los resultados ya están incorporados al notebook y pueden leerse sin ejecutar.

```powershell
python -m pip install -r requirements.txt
python run_experimento.py
python verificar_entrega.py
```

El notebook también es autónomo para Colab: instalará las dependencias si faltan y descargará las seis escenas y el checkpoint. La ejecución en Colab no se ha probado en esta revisión. Si el entorno ya tiene todas las bibliotecas, se conservan sus versiones y se registran en `procedencia.json`.

`auditar_resultados.py` reproduce los diagnósticos y figuras de errores con los datos descargados. `generar_informe.py` genera el PDF con ReportLab y las fuentes Arial de Windows; requiere `reportlab`. Este generador se comprobó con el runtime de documentos de Codex y no es necesario para ejecutar el experimento.

## Integrantes y contribución

Bauz Villa, Juan; Cera Moran, Daniel; Galvis Pajaro, Carlos; Jimenez Atencia, Cher; Mercado Martinez, Sebastian; Molina Vega, Dayana; Romero Pemberthy, Juan; Ruiz Espitia, Kevin; Vargas Becerra, Natalia.

Todos trabajaron de manera colaborativa, según la declaración del grupo. Se declara asistencia de IA en el notebook y el informe. No se asignan aportes individuales no confirmados.
