from fastapi import FastAPI
from pydantic import BaseModel
import random
from transformers import pipeline

# Inicializar la app de FastAPI
app = FastAPI()

# Cargar modelos de Hugging Face
languageDetectionPipeline = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
spellingCorrectionPipeline = pipeline("text2text-generation", model="oliverguhr/spelling-correction-english")

# Modelo para la conversión de temperatura
class TemperatureConversion(BaseModel):
    value: float
    unit: str  # "C" para Celsius, "F" para Fahrenheit

# Modelo para corrección de texto
class TextCorrection(BaseModel):
    text: str

# Endpoint 1: Obtener un dato curioso aleatorio
@app.get("/dato_curioso")
def dato_curioso():
    datos = [
        "Los pulpos tienen tres corazones.",
        "Las abejas pueden reconocer rostros humanos.",
        "El agua caliente se congela más rápido que el agua fría.",
        "Los flamingos nacen con plumas grises, no rosadas.",
        "Los tiburones han existido por más de 400 millones de años.",
        "Tu cerebro genera suficiente electricidad para encender una bombilla pequeña",
        "Tus pulmones tienen una superficie total de aproximadamente el tamaño de una cancha de tenis",
        "Las uñas de las manos crecen más rápido que las de los pies"
    ]
    return {"dato_curioso": random.choice(datos)}

# Endpoint 2: Conversión de temperatura
@app.post("/convertir_temperatura")
def convertir_temperatura(temp: TemperatureConversion):
    if temp.unit.upper() == "C":
        resultado = (temp.value * 9/5) + 32
        return {"Fahrenheit": resultado}
    elif temp.unit.upper() == "F":
        resultado = (temp.value - 32) * 5/9
        return {"Celsius": resultado}
    else:
        return {"error": "Unidad no válida. Usa 'C' para Celsius o 'F' para Fahrenheit."}

# Endpoint 3 con HF: Detección de idioma
@app.post("/detectar_idioma")
def detectar_idioma(texto: TextCorrection):
    resultado = languageDetectionPipeline(texto.text)
    return {"idioma_detectado": resultado[0]['label']}

# Endpoint 4 con HF: Corrección ortográfica en inglés
@app.post("/corregir_texto")
def corregir_texto(texto: TextCorrection):
    resultado = spellingCorrectionPipeline(texto.text)
    return {"texto_corregido": resultado[0]["generated_text"]}

# Endpoint 5: Generación de nombres aleatorios a partir del género
@app.get("/generar_nombre")
def generar_nombre(genero: str):
    nombres_masculinos = ["Carlos", "Luis", "Andrés", "Fernando", "Juan", "José"]
    nombres_femeninos = ["María", "Elisa", "Carmen", "Isabel", "Ana", "Lucía"]
    
    if genero.lower() == "m":
        return {"nombre_generado": random.choice(nombres_masculinos)}
    elif genero.lower() == "f":
        return {"nombre_generado": random.choice(nombres_femeninos)}
    else:
        return {"error": "Género no válido. Usa 'masculino' o 'femenino'."}
