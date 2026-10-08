"""
(Parte 2)
Transformación de textos cortos 
Implemente la técnica de Bag of Words (BoW) para obtener la representación vectorial de cada documento 
Por frecuencia 
Por presencia 
Agregue una columna al final que contendrá la clase a la que pertence el documento

"""
import os
import re
import numpy as np
import pandas as pd

# Ruta de la carpeta que contiene los archivos .txt
carpeta_textos = r"E:\MineriaDts\TRS\PYT\Textos_R1"

# Ruta del archivo CSV con las clases
archivo_clases = r"E:\MineriaDts\TRS\PYT\Textos_R1\textos_cortos.csv"

# Lista básica de stopwords (puedes ampliarla si deseas)
stopwords = {
    "y", "de", "la", "al", "el", "en", "a", "los", "las", "un", "una", "que", "con", "por", "para",
    "o", "se", "del", "lo", "le", "su", "sus", "como", "más", "ya", "pero", "eso", "este", "esta",
    "estos", "estas", "hay", "si", "no", "ni", "es", "son", "fue", "era", "ser", "muy", "también"
}

# Función para limpiar texto (igual que en la Parte 1)
def limpiar_texto(texto):
    """
    Limpia un texto eliminando signos de puntuación, números, palabras alfanuméricas y stopwords.
    Devuelve una lista de palabras limpias.
    """
    # Eliminar palabras con números (alfanuméricos)
    texto = re.sub(r"\b\w*\d\w*\b", "", texto)
    # Eliminar signos de puntuación
    texto = re.sub(r"[^\w\s]", "", texto)
    # Convertir a minúsculas
    texto = texto.lower()
    # Tokenizar y eliminar stopwords
    palabras = texto.split()
    palabras_limpias = [palabra for palabra in palabras if palabra not in stopwords]
    return palabras_limpias

# Obtener vocabulario único
def obtener_vocabulario(carpeta):
    vocabulario = set()
    if not os.path.exists(carpeta):
        print(f"Error: La carpeta '{carpeta}' no existe.")
        return vocabulario

    for archivo in os.listdir(carpeta):
        if archivo.endswith(".txt"):
            ruta_archivo = os.path.join(carpeta, archivo)
            try:
                with open(ruta_archivo, "r", encoding="utf-8") as f:
                    texto = f.read()
            except UnicodeDecodeError:
                with open(ruta_archivo, "r", encoding="latin-1") as f:
                    texto = f.read()

            palabras_limpias = limpiar_texto(texto)
            vocabulario.update(palabras_limpias)

    return vocabulario

# Cargar clases desde CSV
def cargar_clases(ruta_csv):
    df = pd.read_csv(ruta_csv)
    clases = dict(zip(df['Archivo'], df['clase']))
    return clases

# Generar BoW (frecuencia o presencia)
def generar_bow(carpeta, vocabulario, clases, modo="frecuencia"):
    matriz = []
    nombres_archivos = []
    clases_documentos = []

    for archivo in os.listdir(carpeta):
        if archivo.endswith(".txt"):
            ruta_archivo = os.path.join(carpeta, archivo)
            try:
                with open(ruta_archivo, "r", encoding="utf-8") as f:
                    texto = f.read()
            except UnicodeDecodeError:
                with open(ruta_archivo, "r", encoding="latin-1") as f:
                    texto = f.read()

            palabras_limpias = limpiar_texto(texto)
            vector = []

            for palabra in vocabulario:
                if modo == "frecuencia":
                    vector.append(palabras_limpias.count(palabra))
                elif modo == "presencia":
                    vector.append(1 if palabra in palabras_limpias else 0)

            matriz.append(vector)
            nombres_archivos.append(archivo)
            clases_documentos.append(clases.get(archivo, -1))  # Se busca la clase por nombre de archivo

    return np.array(matriz), nombres_archivos, clases_documentos

# Guardar matriz BoW en CSV
def guardar_bow(matriz, nombres_archivos, clases_documentos, vocabulario, modo, carpeta):
    columnas = list(vocabulario) + ["Clase"]
    datos = np.hstack((matriz, np.array(clases_documentos).reshape(-1, 1)))
    df = pd.DataFrame(datos, columns=columnas, index=nombres_archivos)
    ruta_salida = os.path.join(carpeta, f"BoW_{modo}.csv")
    df.to_csv(ruta_salida, encoding="utf-8")
    print(f"BoW ({modo}) guardado en: {ruta_salida}")

# --------------------------
# EJECUCIÓN FINAL
# --------------------------

# Obtener vocabulario de la carpeta
vocabulario = obtener_vocabulario(carpeta_textos)

# Cargar las clases desde CSV
clases = cargar_clases(archivo_clases)

# BoW por frecuencia
matriz_frecuencia, nombres_archivos, clases_documentos = generar_bow(carpeta_textos, vocabulario, clases, modo="frecuencia")
guardar_bow(matriz_frecuencia, nombres_archivos, clases_documentos, vocabulario, "frecuencia", carpeta_textos)

# BoW por presencia
matriz_presencia, _, _ = generar_bow(carpeta_textos, vocabulario, clases, modo="presencia")
guardar_bow(matriz_presencia, nombres_archivos, clases_documentos, vocabulario, "presencia", carpeta_textos)
