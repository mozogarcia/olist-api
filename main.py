import os
# Forzar a las librerías a usar solo 1 hilo para no saturar la RAM de Render
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from preparacion import crear_variables

CARPETA = Path(__file__).resolve().parent

# Cargar modelo y umbral
modelo = joblib.load(CARPETA / "modelo_riesgo.joblib")

with open(CARPETA / "umbral_riesgo.json", "r", encoding="utf-8") as archivo:
    umbral = json.load(archivo)["umbral_riesgo"]

app = FastAPI(title="API de riesgo de malas reseñas")

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
    return {"mensaje": "API de riesgo de malas reseñas activa", "estado": "ok"}

@app.post("/riesgo")
def predecir_riesgo(pedido: PedidoContrato):
    try:
        # Convertir a DataFrame procesando 1 solo registro
        datos = pd.DataFrame([pedido.model_dump()])
        variables = crear_variables(datos)

        # Inferencia
        probabilidad = float(modelo.predict_proba(variables)[0, 1])

        # Limpieza activa e inmediata de RAM
        del datos, variables
        gc.collect()

        return {
            "probabilidad": round(probabilidad, 4),
            "riesgo_alto": bool(probabilidad >= umbral),
            "umbral_riesgo": round(float(umbral), 4)
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {str(error)}")
