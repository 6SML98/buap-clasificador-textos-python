"""
(Parte 3)
Implementación de Naive Bayes Classifier 
Implemente el entrenamiento de NBC 
1) Variante producto de probabilidades 
2) Variante de suma de logaritmos 
Implemente las funciones de prediccion: 
Predict(x) retorna la prediccion de cada instancia 
x es una matriz de tamaño num_instancias x num_atributos. 
predict_probabilities(x): retorna las probabilidades de cada clase 
predict_logScores(x): retorna los puntajes de la suma de logaritmos. 

"""
import os
import csv
import string
import random
import math
from collections import Counter
from math import log2


# ========== STOPWORODS ==========
CSV_RUTA = "textos_cortos.csv"
CARPETA_TEXTOS = r"C:\Users\carmi\OneDrive\Documentos\7 semestre\Mineria\Nueva carpeta\Textos"
STOPWORDS = [
    'el', 'la', 'los', 'las', 'de', 'del', 'y', 'en', 'a', 'que',
    'es', 'un', 'una', 'por', 'con', 'para', 'no', 'se', 'al', 'lo',
    'como', 'su', 'si', 'pero', 'más', 'o', 'ya', 'ha', 'porque', 'muy'
]

# ========== FUNCIONES DE TEXTO ==========
def limpiar_texto(texto):
    texto = texto.lower()
    texto = ''.join([c if c in string.ascii_letters + " áéíóúñ" else ' ' for c in texto])
    palabras = texto.split()
    palabras = [p for p in palabras if not any(c.isdigit() for c in p)]
    palabras = [p for p in palabras if p not in STOPWORDS]
    return palabras

# ========== ETAPA 1: LIMPIEZA Y VOCABUALRIO==========
vocabulario = set()
textos_limpios = []

with open(CSV_RUTA, "r", encoding="utf-8") as archivo_csv:
    lector = csv.reader(archivo_csv)
    next(lector)

    for fila in lector:
        ruta_archivo, clase = fila[0], fila[1]
        nombre_archivo = os.path.basename(ruta_archivo).lower().replace("archivos", "archivo")
        ruta_completa = os.path.join(CARPETA_TEXTOS, nombre_archivo)

        try:
            with open(ruta_completa, "r", encoding="utf-8", errors="ignore") as f:
                contenido = f.read()
        except FileNotFoundError:
            print(f" Archivo no encontrado: {ruta_completa}")
            continue

        palabras = limpiar_texto(contenido)
        vocabulario.update(palabras)
        textos_limpios.append((palabras, int(clase)))

vocabulario_ordenado = sorted(vocabulario)
vocab_index = {palabra: i for i, palabra in enumerate(vocabulario_ordenado)}

# Guardar vocabulario
with open("vocabulario.txt", "w", encoding="utf-8") as vocab_file:
    for palabra in vocabulario_ordenado:
        vocab_file.write(palabra + "\n")

# Generar vectores BoW
datos_bow = []
for palabras, clase in textos_limpios:
    vector_frec = [0] * len(vocabulario_ordenado)
    for palabra in palabras:
        if palabra in vocab_index:
            vector_frec[vocab_index[palabra]] += 1
    vector_pres = [1 if count > 0 else 0 for count in vector_frec]
    datos_bow.append((vector_frec + [clase], vector_pres + [clase]))

# Guardar BoW
encabezado = vocabulario_ordenado + ["clase"]
with open("bow_frecuencia.csv", "w", encoding="utf-8", newline='') as f_frec, \
     open("bow_presencia.csv", "w", encoding="utf-8", newline='') as f_pres:
    writer_frec = csv.writer(f_frec)
    writer_pres = csv.writer(f_pres)
    writer_frec.writerow(encabezado)
    writer_pres.writerow(encabezado)
    for vec_frec, vec_pres in datos_bow:
        writer_frec.writerow(vec_frec)
        writer_pres.writerow(vec_pres)

# ========== ETAPA 2: DIVISIÓN ==========
def mostrar_distribucion(nombre, filas):
    clases = [fila[-1] for fila in filas]
    conteo = Counter(clases)
    print(f"\n Distribución de clases en {nombre}:")
    for clase, cantidad in sorted(conteo.items()):
        print(f"  Clase {clase} → {cantidad} documentos")

def dividir_train_test(archivo_entrada, nombre_salida):
    with open(archivo_entrada, "r", encoding="utf-8") as f:
        lector = list(csv.reader(f))
        encabezado = lector[0]
        filas = lector[1:]

    random.shuffle(filas)
    corte = int(0.8 * len(filas))
    train = filas[:corte]
    test = filas[corte:]

    with open(f"{nombre_salida}_train.csv", "w", encoding="utf-8", newline="") as f_train, \
         open(f"{nombre_salida}_test.csv", "w", encoding="utf-8", newline="") as f_test:
        writer_train = csv.writer(f_train)
        writer_test = csv.writer(f_test)
        writer_train.writerow(encabezado)
        writer_test.writerow(encabezado)
        writer_train.writerows(train)
        writer_test.writerows(test)

    mostrar_distribucion(f"{nombre_salida}_train.csv", train)
    mostrar_distribucion(f"{nombre_salida}_test.csv", test)

dividir_train_test("bow_presencia.csv", "presencia")
#dividir_train_test("bow_frecuencia.csv", "frecuencia")

# ========== ETAPA 3: NAIVE BAYES ==========
def cargar_datos(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        lector = list(csv.reader(f))
        encabezado = lector[0]
        datos = lector[1:]
        X = [list(map(int, fila[:-1])) for fila in datos]
        y = [int(fila[-1]) for fila in datos]
    return X, y, encabezado[:-1]

class NaiveBayesClassifier:
    def _init_(self):
        self.priors = {}
        self.likelihoods = {}
        self.clases = set()
        self.vocab_size = 0

    def entrenar(self, X, y):
        self.clases = set(y)
        self.vocab_size = len(X[0])
        total_docs = len(y)
        conteo_clases = {}
        conteo_palabras = {}

        for i in range(total_docs):
            clase = y[i]
            if clase not in conteo_clases:
                conteo_clases[clase] = 0
                conteo_palabras[clase] = [0] * self.vocab_size
            conteo_clases[clase] += 1
            for j in range(self.vocab_size):
                conteo_palabras[clase][j] += X[i][j]

        self.priors = {c: conteo_clases[c]/total_docs for c in self.clases}
        self.likelihoods = {}
        for c in self.clases:
            total_palabras = sum(conteo_palabras[c])
            self.likelihoods[c] = [
                (conteo_palabras[c][i] + 1) / (total_palabras + self.vocab_size)
                for i in range(self.vocab_size)
            ]

    def predict(self, X):
        return [self._predecir_una(x) for x in X]

    def _predecir_una(self, x):
        log_scores = self.predict_logScores([x])[0]
        return max(log_scores, key=log_scores.get)

    def predict_probabilities(self, X):
        resultados = []
        for x in X:
            scores = {}
            for c in self.clases:
                prob = self.priors[c]
                for i in range(self.vocab_size):
                    prob *= self.likelihoods[c][i] ** x[i]
                scores[c] = prob
            total = sum(scores.values())
            probabilidades = {c: scores[c]/total if total > 0 else 0 for c in self.clases}
            resultados.append(probabilidades)
        return resultados

    def predict_logScores(self, X):
        resultados = []
        for x in X:
            log_scores = {}
            for c in self.clases:
                log_prob = math.log2(self.priors[c])
                for i in range(self.vocab_size):
                    if x[i] > 0:
                        log_prob += x[i] * math.log2(self.likelihoods[c][i])
                log_scores[c] = log_prob
            resultados.append(log_scores)
        return resultados

# Cargar datos de entrenamiento y prueba
#X_train, y_train, _ = cargar_datos("frecuencia_train.csv")
#X_test, y_test, _ = cargar_datos("frecuencia_test.csv")
X_train, y_train, _ = cargar_datos("presencia_train.csv")
X_test, y_test, _ = cargar_datos("presencia_test.csv")


modelo = NaiveBayesClassifier()
modelo.entrenar(X_train, y_train)

predicciones = modelo.predict(X_test)
probabilidades = modelo.predict_probabilities(X_test)
log_scores = modelo.predict_logScores(X_test)

accuracy = sum(1 for i in range(len(y_test)) if predicciones[i] == y_test[i]) / len(y_test)
print(f"\n Exactitud del clasificador Naive Bayes: {accuracy:.2%}")
print(f"\n Porcentaje de predicciones correctas usando predict(x) con matriz de tamaño {len(X_test)}x{len(X_test[0])}: {accuracy:.2%}")

print("\n Primeras 5 predicciones:")
for i in range(min(5, len(y_test))):
    print(f"- Real: {y_test[i]}, Predicho: {predicciones[i]}")
    print(f"  Probabilidades: {probabilidades[i]}")
    print(f"  LogScores: {log_scores[i]}\n")

# Guardar resultados
with open("resultados_presencia.csv", "w", encoding="utf-8", newline="") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["real", "predicho", "prob_clase_0", "prob_clase_1", "prob_clase_2"])
    for i in range(len(y_test)):
        probs = probabilidades[i]
        writer.writerow([
            y_test[i],
            predicciones[i],
            round(probs.get(0, 0), 6),
            round(probs.get(1, 0), 6),
            round(probs.get(2, 0), 6)
        ])

print(" Archivo generado: resultados_presencia.csv")
# Leer resultados.csv y calcular efectividad final
def calcular_efectividad_desde_csv(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        lector = list(csv.reader(f))
        datos = lector[1:]  # omitir encabezado
        total = len(datos)
        correctos = sum(1 for fila in datos if fila[0] == fila[1])
        efectividad = correctos / total if total > 0 else 0
        print(f" Porcentaje de efectividad desde resultados.csv: {efectividad:.2%}")

calcular_efectividad_desde_csv("resultados_presencia.csv")