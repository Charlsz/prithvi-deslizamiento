# Grupo 2 — Detección de deslizamientos de tierra con Prithvi

**Curso:** Inteligencia Artificial  
**Profesor:** Eduardo Enrique Zurek Varela  
**Fecha de entrega:** 23 de noviembre de 2026  
**Fecha de ejecución verificada:** 17 de noviembre de 2026  
**Repositorio:** https://github.com/Charlsz/prithvi-deslizamiento

## Integrantes y aportes

| Integrante | Aporte principal |
| --- | --- |
| Bauz Villa, Juan | Revisión de la ficha del grupo, formulación del problema y apoyo en el marco teórico |
| Cera Moran, Daniel | Documentación de datos, canales, leyenda de clases y partición train/test |
| Galvis Pajaro, Carlos | Ejecución del experimento, corrección de carga de Prithvi, organización del repositorio y verificación de entregables |
| Jimenez Atencia, Cher | Apoyo en implementación del notebook, registro de procedencia y versiones |
| Mercado Martinez, Sebastian | Evaluación cuantitativa, tablas de métricas y comparación con la línea base |
| Molina Vega, Dayana | Análisis de errores, revisión de figuras y redacción de limitaciones |
| Romero Pemberthy, Juan | Propuesta de transferencia a Colombia y criterios de validación regional |
| Ruiz Espitia, Kevin | Consideraciones éticas, riesgos de uso y declaración de integridad |
| Vargas Becerra, Natalia | Conclusiones, referencias en formato IEEE y revisión final del informe |

## 1. Introducción

En este trabajo buscamos identificar áreas afectadas por deslizamientos de tierra a partir de imágenes satelitales. La pregunta es qué tan bien las características extraídas por Prithvi permiten separar las regiones afectadas del resto del terreno. No buscamos predecir cuándo va a ocurrir un deslizamiento, sino reconocer las zonas que ya están señaladas en las imágenes de referencia.

## 2. Objetivo

Evaluar un encoder Prithvi congelado junto con un clasificador ligero por parches para segmentar deslizamientos y comparar sus resultados con una línea base constante calculada solo a partir del entrenamiento.

## 3. Datos y metodología

Partimos de Landslide4Sense mediante la adaptación del notebook docente. Usamos cuatro muestras de entrenamiento y dos de prueba, seleccionadas por orden de la fuente y no por desempeño. La lectura toma seis canales ópticos del archivo original de catorce canales, con índices [1, 2, 3, 8, 10, 11]. Ese mapeo es provisional y debe confirmarse con el profesor antes de interpretaciones más fuertes.

Las escenas originales suelen medir 128 por 128 píxeles. El pipeline las completa a 224 por 224 con relleno edge para alimentar a Prithvi; las posiciones rellenadas se marcan como inválidas y no entran en las métricas. La normalización se calcula solo con entrenamiento. Se cargan los pesos de Prithvi-EO-1.0-100M en la revisión fijada por el docente y se congela el encoder. Después se entrena un Random Forest sobre parches de 16 por 16 píxeles. La predicción se expande al tamaño de la máscara para compararla con la referencia.

La línea base predice siempre la clase más frecuente del entrenamiento. En nuestra ejecución esa clase fue 0 (no deslizamiento).

## 4. Evaluación

Las métricas principales son IoU y F1 de la clase deslizamiento. También reportamos precisión, recall, soporte por clase, matrices de confusión y resultados por escena. No usamos etiquetas de prueba para elegir decisiones de entrenamiento.

### Resultados globales (agregación por píxeles)

| Método | F1 deslizamiento | IoU deslizamiento | Precisión | Recall |
| --- | ---: | ---: | ---: | ---: |
| Prithvi + Random Forest | 0.0 | 0.0 | 0.0 | 0.0 |
| Línea base constante (clase 0) | 0.0 | 0.0 | 0.0 | 0.0 |

### Media por escena (clase deslizamiento)

| Método | IoU | F1 | Precisión | Recall | Escenas definidas |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prithvi + Random Forest | 0.0 | 0.0 | 0.0 | 0.0 | 2 |
| Línea base | 0.0 | 0.0 | 0.0 | 0.0 | 2 |

### Distribución de entrenamiento (píxeles válidos)

| Clase | Píxeles | Fracción |
| --- | ---: | ---: |
| 0 | 64617 | 0.986 |
| 1 | 919 | 0.014 |

### Matrices de confusión globales

Prithvi + Random Forest:

|  | Predicho 0 | Predicho 1 |
| --- | ---: | ---: |
| Real 0 | 32722 | 0 |
| Real 1 | 46 | 0 |

Línea base:

|  | Predicho 0 | Predicho 1 |
| --- | ---: | ---: |
| Real 0 | 32722 | 0 |
| Real 1 | 46 | 0 |

Archivos de evidencia: carpeta `entregables/` (`metricas.csv`, `metricas_media_por_escena.csv`, `distribucion_train.csv`, matrices de confusión y figuras `01_exploracion.png`, `02_resultado_0.png`, `02_resultado_1.png`, `03_comparacion_metricas.png`).

## 5. Análisis de errores

Revisamos tres casos reales:

1. Acierto de fondo. En las dos escenas de prueba casi toda el área válida es clase 0 y el modelo también predice 0. Eso hace que la clase fondo luzca bien, pero no resuelve la tarea de detectar deslizamientos.

2. Falso negativo en la escena de prueba 0. Hay 33 píxeles de deslizamiento en el borde izquierdo de la zona válida, alrededor de la fila 92. La referencia los marca; la predicción no. En la figura de errores aparecen en rojo.

3. Falso negativo en la escena de prueba 1. Hay 13 píxeles positivos en la esquina inferior derecha, alrededor de la fila 121 y la columna 125. Otra vez el modelo los omite. No encontramos falsos positivos en este piloto.

La causa más clara no es un desajuste visual aislado, sino que el entrenamiento por parches nunca recibió la clase 1. A nivel de píxel sí había deslizamientos, pero ningún parche 16×16 llegó al 50 por ciento de píxeles positivos. El máximo observado en entrenamiento fue cerca del 40 por ciento. Por la regla del notebook docente, esos parches quedaron etiquetados como fondo.

## 6. Preguntas de análisis

Detectar un deslizamiento observado significa señalar en la imagen las áreas donde ya se ve el movimiento en masa. Estimar susceptibilidad significa valorar qué tan propenso es un terreno a sufrirlo. Predecir un evento significa estimar si ocurrirá en el futuro y con qué probabilidad. Nuestro trabajo solo hace detección.

Una carretera o una cantera pueden confundirse con un deslizamiento porque ambas pueden mostrar suelo descubierto, bordes irregulares y cambios de color parecidos a una cicatriz.

La elevación y la pendiente aportarían contexto del relieve. Una imagen anterior ayudaría a separar cambios recientes de superficies que ya estaban descubiertas. Para usar esos datos habría que adaptar y validar el contrato de entrada del modelo.

No se puede asumir que un modelo entrenado o probado en una región funcionará igual en otra. Cambian la vegetación, el relieve, el clima, el sensor y la forma de etiquetar. Hacen falta datos locales y pruebas independientes.

## 7. Propuesta de transferencia a Colombia

Proponemos estudiar una región montañosa de Antioquia. Se necesitarían imágenes satelitales compatibles, inventarios de deslizamientos con ubicación y fecha, elevación, pendiente y etiquetas revisadas. Separaríamos entrenamiento, validación y prueba por evento y por región para evitar contaminación entre conjuntos. Si se añaden canales topográficos, primero habría que ajustar y validar la arquitectura de entrada. Evaluaríamos IoU, F1, precisión y recall en eventos no vistos, con revisión experta antes de cualquier uso operativo. Esta parte es una propuesta; no se ejecutó con datos colombianos.

## 8. Consideraciones éticas

Un falso negativo puede ocultar zonas realmente afectadas y retrasar la atención. Un falso positivo puede generar alarmas injustificadas o desviar recursos. Por eso este piloto no debe presentarse como mapa oficial, sistema de alerta temprana ni producto operativo de gestión del riesgo. También hay que respetar licencias de datos y modelos, citar procedencia y declarar el uso de asistencia de IA. Antes de cualquier uso en Colombia haría falta validación local, revisión experta y una partición por evento o región que evite sobreestimar el desempeño.

## 9. Limitaciones

El piloto es pequeño: cuatro escenas de entrenamiento y dos de prueba. La predicción por parches pierde detalle en bordes y en deslizamientos pequeños. El relleno de 128 a 224 introduce contexto artificial alrededor de la escena real. La correspondencia de canales Landslide4Sense con Prithvi debe confirmarse con el profesor. Los resultados no sirven por sí solos para alertas tempranas ni para gestión del riesgo.

## 10. Conclusiones

En la ejecución real, el modelo no superó a la línea base. Ambos obtuvieron IoU 0 y F1 0 en la clase deslizamiento. El Random Forest predijo siempre fondo porque, tras la agregación mayoritaria por parche, la clase positiva desapareció del entrenamiento. Eso explica los 46 falsos negativos de prueba y la ausencia de falsos positivos.

La conclusión responsable es que este piloto demuestra el flujo experimental y una limitación importante del diseño por parches ante clases raras, pero no demuestra que Prithvi detecte bien deslizamientos en general. Para mejorar haría falta más datos, otra forma de etiquetar parches o una cabeza de segmentación de mayor resolución, siempre sin mirar la prueba para elegir el método.

## 11. Referencias (formato IEEE)

[1] D. Szwarcman et al., “Prithvi-EO-2.0: A versatile multi-temporal foundation model for Earth observation applications,” arXiv:2412.02732, 2024. [Online]. Available: https://arxiv.org/abs/2412.02732

[2] NASA IMPACT, “HLS Foundation Models,” GitHub repository. [Online]. Available: https://github.com/NASA-IMPACT/hls-foundation-os

[3] IBM–NASA Geospatial, “Prithvi-EO-1.0-100M,” Hugging Face. [Online]. Available: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M

[4] C. J. Stewart et al., “TerraTorch: The geospatial foundation models toolkit,” arXiv:2503.20563, 2025. [Online]. Available: https://arxiv.org/abs/2503.20563

[5] Landslide4Sense dataset mirror, Hugging Face Datasets. [Online]. Available: https://huggingface.co/datasets/sasmit545/Landslide4sense

[6] E. E. Zurek Varela, “Enunciado general Prithvi aplicado a problemas geoespaciales” y ficha del Grupo 2, Curso de Inteligencia Artificial, 2026.

[7] Referencia comunitaria mencionada por el docente: https://huggingface.co/srivassid/prithvi-landslide (no se asume checkpoint oficial).

**Procedencia de la ejecución:** revisión de Prithvi `7b231fe5b0684e2ddd40c967fbc02e225ef77dc6`; evidencias en `entregables/` y `taller_deslizamientos/resultados/`.

## 12. Integridad y uso de IA

Ejecución: entorno local Windows, CPU, el 17 de noviembre de 2026.  
Se corrigió la carga del YAML plano de Prithvi para poder instanciar el encoder con la revisión fijada.  
Se usó asistencia de IA para organización del notebook, depuración y redacción. El grupo verificó código, métricas, figuras y conclusiones; los valores numéricos no fueron inventados.
