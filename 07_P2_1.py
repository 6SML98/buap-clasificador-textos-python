"""
(Parte2_1)
Division de datos 
Revuelva los datos por fila 
Separe los documentos cortos en dos partes mutuamente excluyentes y colectivamente exhaustivas: 
80% entrenamiento 
20% prueba
"""
import csv
import random
import os

# Archivo de entrada (debe ser un archivo CSV generado previamente)
ENTRADA = r"E:\MineriaDts\TRS\PYT\Textos_R1\BoW_presencia.csv"

# Verificar si el archivo existe
if not os.path.isfile(ENTRADA):
    raise FileNotFoundError(f"El archivo especificado no existe: {ENTRADA}")

# Cargar datos
with open(ENTRADA, "r", encoding="utf-8") as f:
    lector = list(csv.reader(f))
    encabezado = lector[0]
    filas = lector[1:]

# Mezclar aleatoriamente
random.shuffle(filas)

# Dividir 80/20
total = len(filas)
entrenamiento = int(0.8 * total)

datos_train = filas[:entrenamiento]
datos_test = filas[entrenamiento:]

# Rutas de salida
SALIDA_TRAIN = os.path.join(os.path.dirname(ENTRADA), "train.csv")
SALIDA_TEST = os.path.join(os.path.dirname(ENTRADA), "test.csv")

# Guardar archivos
with open(SALIDA_TRAIN, "w", encoding="utf-8", newline="") as f_train, \
     open(SALIDA_TEST, "w", encoding="utf-8", newline="") as f_test:
    
    writer_train = csv.writer(f_train)
    writer_test = csv.writer(f_test)

    writer_train.writerow(encabezado)
    writer_train.writerows(datos_train)

    writer_test.writerow(encabezado)
    writer_test.writerows(datos_test)

print(f"División completada: '{SALIDA_TRAIN}' y '{SALIDA_TEST}' generados.")