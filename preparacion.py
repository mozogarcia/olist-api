# Encabezado: preparar variables para el modelo de riesgo

import numpy as np
import pandas as pd


def crear_variables(datos):
    """
    Transforma las columnas del contrato en variables predictoras.
    La predicción se realiza en el momento de la entrega del pedido.
    """

    datos = datos.copy()

    columnas_requeridas = [
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

    faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in datos.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas del contrato: {faltantes}"
        )

    # Convertir fechas
    columnas_fechas = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for columna in columnas_fechas:
        datos[columna] = pd.to_datetime(
            datos[columna],
            errors="coerce"
        )

    variables = pd.DataFrame(index=datos.index)

    # Variables de tiempo con imputación segura de Nulos (0.0 por defecto si no existe la fecha)
    variables["dias_desde_compra"] = (
        datos["order_delivered_customer_date"]
        - datos["order_purchase_timestamp"]
    ).dt.total_seconds().fillna(0.0) / 86400.0

    variables["horas_hasta_aprobacion"] = (
        datos["order_approved_at"]
        - datos["order_purchase_timestamp"]
    ).dt.total_seconds().fillna(0.0) / 3600.0

    variables["dias_hasta_transportista"] = (
        datos["order_delivered_carrier_date"]
        - datos["order_approved_at"]
    ).dt.total_seconds().fillna(0.0) / 86400.0

    variables["dias_retraso"] = (
        datos["order_delivered_customer_date"]
        - datos["order_estimated_delivery_date"]
    ).dt.total_seconds().fillna(0.0) / 86400.0

    variables["mes_compra"] = (
        datos["order_purchase_timestamp"].dt.month.fillna(1).astype(int)
    )

    variables["dia_semana_compra"] = (
        datos["order_purchase_timestamp"].dt.dayofweek.fillna(0).astype(int)
    )

    # Variables numéricas del pedido (imputación de nulos a 0)
    columnas_numericas = [
        "n_items",
        "n_vendedores",
        "precio_total",
        "flete_total",
        "peso_total_g",
        "fotos_promedio",
        "cuotas_max",
    ]

    for columna in columnas_numericas:
        variables[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce"
        ).fillna(0.0)

    # Variables categóricas
    columnas_categoricas = [
        "estado_cliente",
        "categoria",
        "estado_vendedor",
        "tipo_pago",
    ]

    for columna in columnas_categoricas:
        variables[columna] = (
            datos[columna]
            .fillna("Desconocido")
            .astype(str)
        )

    # Evitar infinitos o NaNs residuales
    variables = variables.replace([np.inf, -np.inf], np.nan).fillna(0)

    return variables
