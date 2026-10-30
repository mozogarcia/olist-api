
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from preparacion import crear_variables

# Carpeta donde están los archivos de la API
CARPETA = Path(__file__).resolve().parent

# Cargar el modelo y el umbral guardados
modelo = joblib.load(CARPETA / "modelo_riesgo.joblib")

with open(CARPETA / "umbral_riesgo.json", "r", encoding="utf-8") as archivo:
    umbral = json.load(archivo)["umbral_riesgo"]

app = FastAPI(
    title="API de riesgo de malas reseñas",
    description="Predice el riesgo de recibir una reseña de 1 o 2 estrellas.",
    version="1.0.0"
)

# Columnas requeridas para preparar las variables
COLUMNAS_REQUERIDAS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "estado_cliente",
    "n_items",
    "n_vendedores",
    "precio_total",
    "flete_total",
    "peso_total_g",
    "fotos_promedio",
    "categoria",
    "estado_vendedor",
    "tipo_pago",
    "cuotas_max",
]


@app.get("/")
def inicio():
    return {
        "mensaje": "API de riesgo de malas reseñas activa",
        "estado": "ok"
    }


@app.post("/riesgo")
def predecir_riesgo(pedido: dict):
    # Verificar que el pedido incluya todas las columnas necesarias
    faltantes = [
        columna
        for columna in COLUMNAS_REQUERIDAS
        if columna not in pedido
    ]

    if faltantes:
        raise HTTPException(
            status_code=422,
            detail=f"Faltan columnas requeridas: {faltantes}"
        )

    try:
        # Convertir el pedido en DataFrame
        datos = pd.DataFrame([pedido])

        # Aplicar la preparación de variables del proyecto
        variables = crear_variables(datos)

        # Calcular la probabilidad de una mala reseña
        probabilidad = float(modelo.predict_proba(variables)[0, 1])

        return {
            "probabilidad": round(probabilidad, 4),
            "riesgo_alto": bool(probabilidad >= umbral),
            "umbral_riesgo": round(float(umbral), 4)
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al calcular el riesgo: {str(error)}"
        )
