"""
(Parte 1)
Limpieza de textos cortos:
-Eliminar singnos de puntuación, numeros y alfanumericos
-Uso de solo minúsculas
-Crear lista de stopwords
-Eliminar stopwords
-Obtener el vocabulario de su colección de textos
Sin utilizar NLS
"""
import os
from pathlib import Path
import re

# Ruta de la carpeta que contiene los archivos .txt
carpeta_textos = os.getenv("TEXTOS_DIR", str(Path(__file__).resolve().parent / 'examples' / 'texts'))

# Lista de stopwords básicas en español (puedes ampliarla)
stopwords = {
    "y", "de", "la", "al", "el", "en", "a", "los", "las", "un", "una", "que", "con", "por", "para",
    "o", "se", "del", "lo", "le", "su", "sus", "como", "más", "ya", "pero", "eso", "este", "esta",
    "estos", "estas", "hay", "si", "no", "ni", "es", "son", "fue", "era", "ser", "muy", "también"
}

def limpiar_texto(texto):
    """
    Limpia un texto eliminando signos de puntuación, números, palabras alfanuméricas
    y stopwords. Devuelve una lista de palabras limpias.
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

def obtener_vocabulario(carpeta):
    """
    Procesa todos los archivos .txt de una carpeta y genera el vocabulario único.
    """
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

def guardar_vocabulario(vocabulario, carpeta):
    """
    Guarda el vocabulario ordenado en un archivo de texto.
    """
    ruta_salida = os.path.join(carpeta, "vocabulario.txt")
    with open(ruta_salida, "w", encoding="utf-8") as f:
        for palabra in sorted(vocabulario):
            f.write(palabra + "\n")
    print(f"Vocabulario guardado en: {ruta_salida}")

# Ejecutar procesamiento y guardar resultado
vocabulario = obtener_vocabulario(carpeta_textos)
salida = os.getenv("OUTPUT_DIR", "resultados")
os.makedirs(salida, exist_ok=True)
guardar_vocabulario(vocabulario, salida)
