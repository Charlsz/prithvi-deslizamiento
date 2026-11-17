# Reproducción de la ampliación

La ampliación conserva la prueba original 4/2 y añade una comparación con 96 escenas de entrenamiento, 24 de validación y 32 de prueba nuevas. Los nombres y la semilla se fijaron antes de leer las etiquetas. La revisión del dataset y del checkpoint es la misma del piloto.

## Inspeccionar los resultados ya calculados

```powershell
python -m pip install -r requirements.txt
python verificar_mejora.py
```

Las métricas principales están en `experimentos/mejora_rf/resultados/comparacion_final.csv`. Las salidas, protocolos previos a test, selecciones de validación, matrices, muestras y hashes se conservan en esa carpeta. Los NPZ contienen las etiquetas y predicciones necesarias para recalcular métricas sin descargar imágenes o pesos. Los scripts no permiten desarrollar nuevamente las cabezas en la carpeta que ya tiene el test evaluado.

## Repetir el experimento con el mismo protocolo

```powershell
python reproducir_mejora.py --workdir reproducciones/corrida_01
```

Este comando crea una carpeta nueva, copia el protocolo fijo y ejecuta el desarrollo y la evaluación con las mismas decisiones. Reutiliza las escenas locales si están disponibles; de lo contrario las descarga desde la revisión fijada. No cambia la selección a la vista de test ni sobrescribe la evaluación original. La ejecución utiliza CPU si no hay GPU y necesita espacio para datos, características y cabezas entrenadas. No se incluyen pesos ni escenas crudas en el paquete de entrega.

El código se distribuye en etapas:

1. `experimento_mejora.py develop`: descarga train y validación, estima normalización con train, extrae embeddings, entrena el RF mayoritario y selecciona su umbral en validación.
2. `desarrollar_cabezas.py`: compara ocupaciones de parche y cabezas por píxel usando solo train/validación; guarda la selección antes de test.
3. `experimento_mejora.py evaluate`: abre las 32 escenas nuevas y evalúa las dos variantes mayoritarias y la constante de entrenamiento.
4. `evaluar_extensiones.py`: aplica los modelos y umbrales ya fijados; no entrena ni busca opciones sobre test.
5. `visualizar_mejora.py`: produce tablas y tres casos de diagnóstico con criterios explícitos.
6. `verificar_mejora.py`: recalcula métricas con scikit-learn y comprueba que la selección congelada y las particiones coincidan con sus hashes.

## Interpretación y relación con el enunciado

El RF mayoritario ampliado conserva la familia técnica del flujo docente, con encoder congelado y clasificador por parches; cambia la cantidad de datos e incorpora validación. La cabeza por píxel y las reglas de ocupación cambian el método de la guía: son extensiones exploratorias que deben identificarse como tales. No se afirma que el docente haya autorizado usarlas como sustituto del flujo suministrado.

La cabeza con Prithvi por píxel obtuvo F1 medio 0.3068 e IoU media 0.2030 en test; el RF mayoritario con umbral validado obtuvo 0.1944 y 0.1323. La mejora sobre cero es real, pero el desempeño sigue siendo moderado. El control óptico supera a la cabeza con Prithvi en IoU media y métricas globales: debe conservarse, sin afirmar una ventaja general de Prithvi.

Se mantiene pendiente la confirmación docente de canales, escala, normalización y pesos. Tampoco puede demostrarse independencia por evento o región con los metadatos disponibles. Para ajustar otra vez después de esta evaluación se requiere una nueva prueba reservada; no debe seguir seleccionándose con las 32 escenas ya examinadas.
