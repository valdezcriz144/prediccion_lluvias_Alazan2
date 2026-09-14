import io
import json
import os
from api_client import get_weather_forecast
import config
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import xgboost as xgb

st.set_page_config(
    page_title="Predicción Hídrica - Central Alazán",
    page_icon="⚡",
    layout="wide",
)

# --- CONFIGURACIÓN DE AUTO-REFRESCO (Cada 5 minutos = 300,000 ms) ---
st_autorefresh(interval=300000, key="datarefresh_5min")


@st.cache_data(ttl=600)
def cargar_datos_historicos():
    """Carga el dataset y calcula el caudal promedio histórico según el día de la semana."""
    if os.path.exists(config.FILE_CONSOLIDADO):
        df = pd.read_excel(config.FILE_CONSOLIDADO)
        col_q = next(
            (
                c
                for c in df.columns
                if any(
                    k in c.lower() for k in ["caudal_tp", "tp", "turbinado", "caudal"]
                )
            ),
            None,
        )
        col_fecha = next(
            (
                c
                for c in df.columns
                if any(k in c.lower() for k in ["fecha_hora", "fecha", "time"])
            ),
            None,
        )
        if col_q and col_fecha:
            df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce")
            df["dia_semana"] = df[col_fecha].dt.dayofweek
            perfil_dia = df.groupby("dia_semana")[col_q].mean().to_dict()
            return perfil_dia, df[col_q].mean()
    return {}, 3.433


@st.cache_resource
def cargar_modelo():
    """Carga el modelo XGBoost entrenado."""
    if os.path.exists("modelo_alazan.json"):
        model = xgb.XGBRegressor()
        model.load_model("modelo_alazan.json")
        return model
    return None


# --- ENCABEZADO Y TÍTULO ---
st.title("⚡ Predicción de Generación Hídrica - Central Alazán")
st.markdown("---")

perfil_dia, q_promedio_global = cargar_datos_historicos()
modelo = cargar_modelo()

# --- SIDEBAR DE SIMULACIÓN Y EXPORTACIÓN ---
st.sidebar.header("⚙️ Parámetros de Simulación")
horizonte = st.sidebar.number_input(
    "Horizonte de Predicción (Horas):",
    min_value=1,
    max_value=168,
    value=24,
    step=1,
)

# --- OBTENCIÓN DE DATOS METEOROLÓGICOS Y PREDICCIÓN ---
try:
    df_clima = get_weather_forecast()

    df_pred = df_clima.head(horizonte).copy()
    lluvia_max = df_pred["lluvia_mm"].max()

    df_pred["fecha_hora_dt"] = pd.to_datetime(df_pred["fecha_hora"])
    df_pred["dia_semana_idx"] = df_pred["fecha_hora_dt"].dt.dayofweek
    df_pred["q_base_historico"] = (
        df_pred["dia_semana_idx"].map(perfil_dia).fillna(q_promedio_global)
    )

    # --- DESPLAZAMIENTO EN PRECIPITACIÓN Y SUAVIZADO DE INERCIA ---
    df_pred["lluvia_mm_desplazada"] = df_pred["lluvia_mm"].shift(1, fill_value=0.0)

    # 1. Caudal bruto preliminar
    caudal_bruto = np.clip(
        df_pred["q_base_historico"] + (df_pred["lluvia_mm_desplazada"] * 0.38),
        0.0,
        config.CAUDAL_MAX_DISEÑO,
    )

    # 2. Suavizado exponencial para amortiguar saltos bruscos entre horas (Inercia hidrológica)
    df_pred["caudal_estimado"] = caudal_bruto.ewm(span=3, adjust=False).mean()

    # Cálculo de potencia descontando caudal ecológico e interpolando con curva SCADA
    potencias = []
    caudales_turbinados = []
    for q in df_pred["caudal_estimado"]:
        q_disponible = max(0.0, q - 0.01)
        q_turbinado = min(q_disponible, config.CAUDAL_MAX_DISEÑO)
        caudales_turbinados.append(q_turbinado)

        if q_turbinado <= 0:
            potencias.append(0.0)
        else:
            pot_mw = np.interp(
                q_turbinado, config.TABLA_CAUDAL_SCADA, config.TABLA_POTENCIA_SCADA
            )
            pot_mw = min(pot_mw, config.POTENCIA_MAX_PERMITIDA_MW)
            potencias.append(pot_mw)

    df_pred["caudal_turbinado_m3s"] = caudales_turbinados
    df_pred["potencia_estimada_mw"] = potencias

    q_max = df_pred["caudal_estimado"].max()
    pot_max = df_pred["potencia_estimada_mw"].max()
    q_promedio_horizonte = df_pred["q_base_historico"].mean()

    # --- SINCRONIZACIÓN DE HORA ACTUAL (ECUADOR UTC-5) ---
    ahora_ec = pd.Timestamp.utcnow() - pd.Timedelta(hours=5)
    df_hora_actual = df_pred[df_pred["fecha_hora_dt"].dt.hour == ahora_ec.hour]

    if not df_hora_actual.empty:
        pot_actual = df_hora_actual["potencia_estimada_mw"].iloc[0]
        hora_actual_str = pd.to_datetime(df_hora_actual["fecha_hora"].iloc[0]).strftime("%H:00")
    else:
        pot_actual = df_pred["potencia_estimada_mw"].iloc[0]
        hora_actual_str = pd.to_datetime(df_pred["fecha_hora"].iloc[0]).strftime("%H:00")

    # --- EXPORTACIÓN A EXCEL ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Exportar Resultados")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_pred.to_excel(writer, index=False, sheet_name="Predicciones_Alazan")
    buffer.seek(0)

    st.sidebar.download_button(
        label="📊 Descargar Predicción a Excel",
        data=buffer,
        file_name="prediccion_hidrica_alazan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # --- TARJETAS DE MÉTRICAS ---
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Caudal Prom. Diario (Matriz)",
            value=f"{q_promedio_horizonte:.2f} m³/s",
        )

    with col2:
        st.metric(label="Precipitación Máx. API", value=f"{lluvia_max:.2f} mm/h")

    with col3:
        st.metric(label="Caudal Máx. Captado", value=f"{q_max:.3f} m³/s")

    with col4:
        st.metric(
            label=f"Potencia Hora ({hora_actual_str})", 
            value=f"{pot_actual:.3f} MW"
        )

    with col5:
        st.metric(label="Potencia Máx. Proyectada", value=f"{pot_max:.3f} MW")

    st.markdown("---")

    # --- GRÁFICAS ---
    st.subheader("📈 Proyección Hidrológica y de Generación")

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        fig_pot = px.line(
            df_pred,
            x="fecha_hora",
            y="potencia_estimada_mw",
            title="Generación Estimada (MW)",
            labels={
                "fecha_hora": "Fecha / Hora",
                "potencia_estimada_mw": "Potencia Estimada (MW)",
            },
            markers=True,
        )
        fig_pot.update_traces(line_color="#1f77b4")
        fig_pot.update_layout(yaxis_range=[0, 7.5])
        st.plotly_chart(fig_pot, use_container_width=True)

    with g_col2:
        fig_q = px.line(
            df_pred,
            x="fecha_hora",
            y=["caudal_estimado", "caudal_turbinado_m3s"],
            title="Caudal Captado y Turbinado (m³/s)",
            labels={
                "fecha_hora": "Fecha / Hora",
                "value": "Caudal (m³/s)",
                "variable": "Flujo",
            },
            markers=True,
        )
        fig_q.update_layout(yaxis_range=[0, 4.5])
        st.plotly_chart(fig_q, use_container_width=True)

    # --- TABLA DETALLE ---
    st.markdown("---")
    st.subheader("📋 Detalle Horario de Potencia Proyectada")

    df_horario = df_pred[[
        "fecha_hora", 
        "lluvia_mm", 
        "caudal_estimado", 
        "caudal_turbinado_m3s", 
        "potencia_estimada_mw"
    ]].copy()

    df_horario["fecha_hora"] = pd.to_datetime(df_horario["fecha_hora"]).dt.strftime("%Y-%m-%d %H:00")

    df_horario.columns = [
        "Fecha / Hora", 
        "Precipitación (mm/h)", 
        "Caudal Captado (m³/s)", 
        "Caudal Turbinado (m³/s)", 
        "Potencia Generada (MW)"
    ]

    st.dataframe(
        df_horario.style.format({
            "Precipitación (mm/h)": "{:.2f}",
            "Caudal Captado (m³/s)": "{:.3f}",
            "Caudal Turbinado (m³/s)": "{:.3f}",
            "Potencia Generada (MW)": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

except Exception as e:
    st.error(f"Error al cargar las predicciones: {e}")
