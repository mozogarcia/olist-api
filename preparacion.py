import numpy as np
import pandas as pd


def crear_variables(datos: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma las columnas del contrato en variables predictoras limpias.
    Garantiza la ausencia de NaNs e infinitos.
    """
    datos = datos.copy()

    # Convertir fechas a datetime
    columnas_fechas = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for col in columnas_fechas:
        if col in datos.columns:
            datos[col] = pd.to_datetime(datos[col], errors="coerce")

    variables = pd.DataFrame(index=datos.index)

    # Variables de tiempo con imputación por defecto (0.0 si la fecha es nula)
    variables["dias_desde_compra"] = (
        datos["order_delivered_customer_date"] - datos["order_purchase_timestamp"]
    ).dt.total_seconds().fillna(0.0) / 86400.0

    variables["horas_hasta_aprobacion"] = (
        datos["order_approved_at"] - datos["order_purchase_timestamp"]
    ).dt.total_seconds().fillna(0.0) / 3600.0

    variables["dias_hasta_transportista"] = (
        datos["order_delivered_carrier_date"] - datos["order_approved_at"]
    ).dt.total_seconds().fillna(0.0) / 86400.0

    variables["dias_retraso"] = (
        datos["order_delivered_customer_date"] - datos["order_estimated_delivery_date"]
    ).dt.total_seconds().fillna(0.0) / 86400.0

    # Mes y día de la semana
    variables["mes_compra"] = (
        datos["order_purchase_timestamp"].dt.month.fillna(1).astype(int)
    )

    variables["dia_semana_compra"] = (
        datos["order_purchase_timestamp"].dt.dayofweek.fillna(0).astype(int)
    )

    # Variables numéricas
    columnas_numericas = [
        "n_items",
        "n_vendedores",
        "precio_total",
        "flete_total",
        "peso_total_g",
        "fotos_promedio",
        "cuotas_max",
    ]

    for col in columnas_numericas:
        if col in datos.columns:
            variables[col] = pd.to_numeric(datos[col], errors="coerce").fillna(0.0)
        else:
            variables[col] = 0.0

    # Variables categóricas
    columnas_categoricas = [
        "estado_cliente",
        "categoria",
        "estado_vendedor",
        "tipo_pago",
    ]

    for col in columnas_categoricas:
        if col in datos.columns:
            variables[col] = datos[col].fillna("Desconocido").astype(str)
        else:
            variables[col] = "Desconocido"

    # Reemplazar infinitos y nulos residuales
    variables = variables.replace([np.inf, -np.inf], np.nan).fillna(0)

    return variables
