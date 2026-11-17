# Revisión de la entrega del Grupo 2

Revisión del 17 de noviembre de 2026 contra el enunciado general, la ficha específica y el notebook guía. Este documento es una guía para el grupo; el informe y el notebook contienen el trabajo académico. No predice la calificación.

## Correspondencia con la rúbrica

| Criterio del profesor | Evidencia preparada | Estado y límite |
| --- | --- | --- |
| Fundamentos y aplicación, 15 puntos | Marco teórico del notebook e informe, secciones 2 y 3; Prithvi 1.0, MAE, atención, transferencia y detección | Documentado; cada integrante debe poder explicarlo |
| Datos y reproducibilidad, 20 puntos | Sección 4, inventario por muestra, SHA256, normalización, revisiones fijas y versiones | Comprobado localmente; falta confirmación docente del contrato de entrada |
| Implementación y protocolo, 20 puntos | Notebook ejecutado en orden con un kernel nuevo, encoder congelado y cabeza RF; máscara válida común | Ejecutado; conserva el piloto 4/2 de la guía, sin ajustar sobre test |
| Evaluación cuantitativa, 20 puntos | CSV por clase y escena, media por escena, global separado, matrices y verificador independiente | Coherente con las predicciones; IoU y F1 positivos son 0 |
| Interpretación y errores, 15 puntos | Sección 7, tres casos trazables, mapas y acercamientos con coordenadas | Se documentan TN y dos FN; no existen FP ni TP |
| Transferencia y ética, 10 puntos | Secciones 8 y 9, propuesta para Antioquia, separación por evento, datos auxiliares y riesgos | Propuesta, sin afirmar validación colombiana |

## Cambios realizados

- Se eliminó la dependencia de resultados previos al comienzo del notebook; ahora puede ejecutarse con una sesión nueva.
- Se registraron salidas reales con números de ejecución secuenciales.
- Se fijó la revisión de los datos y se corrigió la escritura JSON a UTF-8.
- Se añadió la publicación primaria de Prithvi 1.0 y se distinguió de la lectura sobre Prithvi 2.0.
- Se documentaron las convenciones de métricas sin denominador y el soporte de cada clase.
- Se guardaron las predicciones y etiquetas NPZ en `entregables/` para verificar los CSV.
- Se reemplazaron las comprobaciones por palabras clave por recálculo numérico con scikit-learn, revisión del notebook y casos límite de métricas.
- Se documentó la desaparición de positivos al etiquetar parches: 256 objetivos de fondo y máximo de ocupación positiva 0.3984375.
- Se mejoraron las figuras con contraste, contornos, leyendas y acercamientos; no se generaron imágenes ni etiquetas artificiales.
- Se sustituyeron las asignaciones individuales no confirmadas por la declaración colaborativa proporcionada por el grupo.

## Pendientes antes de presentar

1. Obtener la confirmación docente exigida por la ficha: orden de canales (en particular el canal seleccionado como NIR), escala, normalización y pesos. El profesor no ha enviado aclaraciones, según informó el grupo. La descripción pública de las bandas no basta para resolver el mapeo inequívocamente.
2. Publicar la versión revisada del notebook y sus evidencias en el repositorio de GitHub. Esta revisión organiza los archivos locales y no ejecuta un push.
3. Revisar en grupo las contribuciones y poder explicar el proceso. Se dejó la declaración compartida que confirmó el usuario.
4. Completar el video por cuenta del grupo. No fue revisado ni producido en esta tarea.

## Sobre la calidad del resultado

El enunciado no impone un valor mínimo de IoU ni exige superar la línea base para obtener la máxima nota. Evalúa fundamentos, reproducibilidad, protocolo, métricas, interpretación y transferencia. Aun así, este piloto no es un detector con buen desempeño: omite los 46 positivos de prueba. Una presentación clara no cambia esa limitación.

Para buscar mejores métricas habría que preservar ejemplos positivos y validar otra cabeza, otra regla de objetivos o un entrenamiento más representativo. La ficha pide el flujo docente o una alternativa autorizada; por eso no se sustituyó el método por una ruta no confirmada. Tampoco se modificaron las decisiones para favorecer las dos escenas de prueba que ya habían sido examinadas. Cualquier desarrollo posterior debe declarar esa exposición y reservar prueba nueva, además de una validación separada.

El archivo `entregables/revision_entrega.json` certifica únicamente las comprobaciones técnicas realizadas. No certifica aprobación del contrato de entrada, publicación remota, capacidad operativa ni calificación.

## Ampliación ejecutada después de la revisión del piloto

Se añadieron 96 escenas de entrenamiento, 24 de validación y 32 de prueba nuevas, seleccionadas sin mirar etiquetas y con protocolos guardados antes de test. Los resultados y la comparación completa están en la sección 12 del informe y en `experimentos/mejora_rf/resultados/`. Se conserva el piloto original y no se elige el modelo por su puntuación en test.

El RF mayoritario con umbral seleccionado en validación obtiene F1 medio 0.1944 e IoU media 0.1323. La cabeza por píxel con Prithvi y bandas ópticas, elegida con validación, obtiene F1 medio 0.3068 e IoU media 0.2030. El control óptico obtiene mejores métricas globales e IoU media que esa cabeza: la comparación no demuestra superioridad general de Prithvi.

El mayoritario ampliado conserva el método de encoder congelado y RF por parche, con más datos y validación. Las otras reglas de objetivos y la cabeza por píxel se identifican como extensiones exploratorias, sin sustituir el flujo docente. Es necesario confirmar con el profesor si acepta el tamaño ampliado y el uso de la extensión como método principal. También sigue pendiente la confirmación de canales, escala, normalización y pesos.

La mejora elimina el resultado constante, pero no acredita desempeño alto. `verificar_mejora.py` comprueba las métricas y la integridad de la selección sobre los archivos guardados; `REPRODUCIR_MEJORA.md` explica las etapas y la reproducción del mismo protocolo.
