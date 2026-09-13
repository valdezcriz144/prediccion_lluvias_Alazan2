import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import config
from api_client import get_weather_forecast

st.set_page_config(
    page_title="Predicción Hídrica - Central Alazán",
    page_icon="⚡",
    layout="wide"
)

# Autorefresh cada 10 minutos
st_autorefresh(interval=10 * 60 * 1000, key="datarefresh")

@st.cache_resource
def cargar_modelo():
    try:
        modelo = xgb.XGBRegressor()
        modelo.load_model(config.FILE_MODELO)
        return modelo
    except Exception:
        return None

@st.cache_data
def cargar_datos_base():
    try:
        df = pd.read_excel(config.FILE_CONSOLIDADO)
        return df
    except Exception:
        return None

st.title("⚡ Predicción de Generación Hídrica - Central Alazán")

# Sidebar - Parámetros
st.sidebar.header("⚙️ Parámetros de Simulación")
horas_prediccion = st.sidebar.slider("Horizonte de Predicción (Horas):", min_value=6, max_value=72, value=24, step=6)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Estado SCADA Actual")

df_base = cargar_datos_base()
default_q = float(df_base['caudal_prin_m3s'].iloc[-1]) if (df_base is not None and 'caudal_prin_m3s' in df_base.columns) else 2.89

q_scada_actual = st.sidebar.number_input(
    "Caudal Actual SCADA (m³/s):", 
    min_value=0.5, 
    max_value=3.44, 
    value=default_q, 
    step=0.05
)

nivel_tc_actual = st.sidebar.number_input(
    "Nivel Tanque de Carga (m):", 
    min_value=1.0, 
    max_value=5.0, 
    value=4.09, 
    step=0.01
)

# Consulta API Meteorológica
df_clima = get_weather_forecast()

if df_clima is not None and not df_clima.empty:
    # Definir rango: desde hace 3 horas en adelante
    ahora = pd.Timestamp.now().floor('H')
    hace_3_horas = ahora - pd.Timedelta(hours=3)
    
    # Filtrar tomando 3 horas pasadas + el horizonte futuro seleccionado
    df_filtrado = df_clima[df_clima['fecha_hora'] >= hace_3_horas].head(horas_prediccion + 3).copy()
    
    lluvia_vector = df_filtrado['lluvia_mm'].values
    n_puntos = len(lluvia_vector)
    
    # Hidrograma de respuesta para la cuenca de Alazán
    C_escorrentia = 0.42
    pesos_hidrograma = np.array([0.15, 0.40, 0.25, 0.12, 0.08])
    escorrentia = np.convolve(lluvia_vector, pesos_hidrograma, mode='full')[:n_puntos] * C_escorrentia
    
    alpha_recesion = 0.015
    caudal_estimado = np.zeros(n_puntos)
    q_actual = q_scada_actual
    
    for t in range(n_puntos):
        q_base = max(1.20, q_actual * np.exp(-alpha_recesion))
        q_actual = q_base + escorrentia[t]
        caudal_estimado[t] = np.clip(q_actual, 0.50, 3.44)

    # Evaluación de modelo XGBoost o curva electromecánica SCADA
    modelo = cargar_modelo()
    if modelo is not None:
        try:
            X_pred = pd.DataFrame({
                'caudal_prin_m3s': caudal_estimado,
                'nivel_tc_r': np.full(n_puntos, nivel_tc_actual),
                'azud_prin': np.full(n_puntos, 2.62)
            })
            potencia_predicha = modelo.predict(X_pred)
        except Exception:
            potencia_predicha = caudal_estimado * 1.73
    else:
        potencia_predicha = caudal_estimado * 1.73

    potencia_predicha = np.clip(potencia_predicha, 0, 6.23)

    df_filtrado['caudal_estimado'] = caudal_estimado
    df_filtrado['potencia_estimada'] = potencia_predicha

    # Separar ventana de predicción futura para visualización
    df_pronostico = df_filtrado[df_filtrado['fecha_hora'] >= ahora].head(horas_prediccion)
    if df_pronostico.empty:
        df_pronostico = df_filtrado.tail(horas_prediccion)

    # Tarjetas de Métricas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Caudal SCADA Inicial", f"{q_scada_actual:.2f} m³/s")
    col2.metric("Precipitación Máx. (3h + Futuro)", f"{df_filtrado['lluvia_mm'].max():.2f} mm/h")
    col3.metric("Caudal Máx. Proyectado", f"{df_pronostico['caudal_estimado'].max():.2f} m³/s")
    col4.metric("Potencia Máx. Proyectada", f"{df_pronostico['potencia_estimada'].max():.2f} MW")

    st.subheader("📈 Proyección Hidrológica y de Generación")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        fig_mw = px.line(
            df_pronostico, 
            x='fecha_hora', 
            y='potencia_estimada', 
            title="Generación Estimada (MW)",
            labels={'fecha_hora': 'Fecha y Hora', 'potencia_estimada': 'Potencia (MW)'},
            markers=True
        )
        fig_mw.update_traces(line_color='#0066cc', line_width=2.5)
        st.plotly_chart(fig_mw, use_container_width=True)
        
    with col_g2:
        fig_q = px.line(
            df_filtrado, 
            x='fecha_hora', 
            y=['caudal_estimado', 'lluvia_mm'], 
            title="Caudal Estimado (m³/s) vs Lluvia API (Últimas 3h + Pronóstico)",
            labels={'fecha_hora': 'Fecha y Hora', 'value': 'Magnitud', 'variable': 'Parámetro'},
            markers=True
        )
        st.plotly_chart(fig_q, use_container_width=True)

else:
    st.error("No se pudo obtener la información meteorológica desde api_client.py.")
