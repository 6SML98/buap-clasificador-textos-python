# Clasificador de textos

Práctica de minería de datos: limpieza, bag of words, división 80/20 y clasificación Naive Bayes multinomial.

## Requisitos

Python 3. Para 07_P2.py instala numpy y pandas con `python -m pip install -r requirements.txt`.

## Ejecutar

```text
python 07.py
python 07_P2.py
python 07_P2_1.py
python 07_P3.py
python -m unittest discover -s tests -v
```

Usa los 15 documentos sintéticos de examples/ por defecto. TEXTOS_DIR y CLASES_CSV permiten usar tu corpus; OUTPUT_DIR controla la carpeta de salida y RANDOM_SEED la semilla. El CSV lleva Archivo,clase. Los resultados se guardan en resultados/.

## Verificación del 8 de octubre de 2026

Las cuatro etapas terminaron y las pruebas del modelo pasaron, incluyendo documentos largos que antes podían perder sus probabilidades por subdesbordamiento. Se corrigieron rutas locales y el constructor. Los datos de ejemplo prueban ejecución, no calidad predictiva. El flujo académico construye el vocabulario antes de dividir: para evaluar generalización, aprende el vocabulario exclusivamente con el conjunto de entrenamiento.
