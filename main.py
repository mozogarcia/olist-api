import gc
import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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

# Esquema Pydantic para validar y mapear los campos requeridos
class PedidoContrato(BaseModel):
    order_id: Optional[str] = "test_123"
    order_purchase_timestamp: str
    order_approved_at: Optional[str] = None
    order_delivered_carrier_date: Optional[str] = None
    order_delivered_customer_date: Optional[str] = None
    order_estimated_delivery_date: str
    estado_cliente: str
    n_items: int = Field(..., ge=1)
    n_vendedores: int = Field(..., ge=1)
    precio_total: float = Field(..., ge=0.0)
    flete_total: float = Field(..., ge=0.0)
    peso_total_g: Optional[float] = Field(None, ge=0.0)
    fotos_promedio: Optional[float] = Field(None, ge=0.0)
    categoria: Optional[str] = None
    estado_vendedor: str
    tipo_pago: Optional[str] = None
    cuotas_max: Optional[float] = Field(None, ge=0.0)


@app.get("/")
def inicio():
    return {
        "mensaje": "API de riesgo de malas reseñas activa",
        "estado": "ok"
    }


@app.post("/riesgo")
def predecir_riesgo(pedido: PedidoContrato):
    try:
        # Convertir el objeto Pydantic a diccionario y luego a DataFrame
        datos_dict = pedido.model_dump()
        datos = pd.DataFrame([datos_dict])

        # Aplicar la preparación de variables del proyecto
        variables = crear_variables(datos)

        # Calcular la probabilidad de una mala reseña
        prob_array = modelo.predict_proba(variables)
        probabilidad = float(prob_array[0, 1])

        # Liberar memoria para evitar que Render mate el proceso (502 Bad Gateway)
        del datos, variables, datos_dict
        gc.collect()

        return {
            "probabilidad": round(probabilidad, 4),
            "riesgo_alto": bool(probabilidad >= umbral),
            "umbral_riesgo": round(float(umbral), 4)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al calcular el riesgo: {str(error)}"
        )
