import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import config
from api_client import obtener_pronostico_lluvia

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

# Sidebar
st.sidebar.header("⚙️ Parámetros de Simulación")
horas_prediccion = st.sidebar.slider("Horizonte de Predicción (Horas):", min_value=6, max_value=72, value=24, step=6)

# 1. Obtener pronóstico de lluvia de la API en vivo
df_pronostico = obtener_pronostico_lluvia(config.LATITUD, config.LONGITUD, horas_prediccion)

if df_pronostico is not None and not df_pronostico.empty:
    df_base = cargar_datos_base()
    
    # Obtener el último caudal reportado como punto de partida
    if df_base is not None and 'caudal_prin_m3s' in df_base.columns:
        q_inicial = df_base['caudal_prin_m3s'].iloc[-1]
        nivel_inicial = df_base['nivel_tc_r'].iloc[-1] if 'nivel_tc_r' in df_base.columns else 4.08
        azud_inicial = df_base['azud_prin'].iloc[-1] if 'azud_prin' in df_base.columns else 3.0
    else:
        q_inicial = 2.15
        nivel_inicial = 4.08
        azud_inicial = 3.0

    # 2. Extraer precipitaciones del pronóstico API
    lluvia_api = df_pronostico['precipitacion'].values
    
    # 3. Modelo de Hidrograma Unitario para la cuenca de Alazán
    # Coeficiente de escorrentía para cuenca alta andina
    C_escorrentia = 0.42 
    # Ponderación del tiempo de concentración de la cuenca (retardos de 0h, 1h, 2h, 3h, 4h)
    pesos_hidrograma = np.array([0.10, 0.35, 0.30, 0.15, 0.10])
    
    # Convolución para simular el viaje del agua desde la cuenca hasta la captación
    escorrentia_lluvia = np.convolve(lluvia_api, pesos_hidrograma, mode='full')[:horas_prediccion] * C_escorrentia
    
    # Simular variación de caudal (Recesión natural + Impulso de la lluvia API)
    alpha_recesion = 0.025 # Tasa de agotamiento del caudal base sin lluvia
    caudal_estimado = np.zeros(horas_prediccion)
    q_actual = q_inicial
    
    for t in range(horas_prediccion):
        # El caudal recede naturalmente si no hay lluvia, o aumenta con la escorrentía
        q_base = max(1.20, q_actual * np.exp(-alpha_recesion))
        q_actual = q_base + escorrentia_lluvia[t]
        
        # Respetar límites físicos de captación de la Central Alazán
        caudal_max = getattr(config, 'CAUDAL_MAX_DISEÑO', 3.44)
        caudal_eco = getattr(config, 'CAUDAL_ECOLOGICO', 0.50)
        caudal_estimado[t] = np.clip(q_actual, caudal_eco, caudal_max)

    # 4. Cálculo de Potencia Estimada (MW)
    modelo = cargar_modelo()
    if modelo is not None:
        try:
            X_pred = pd.DataFrame({
                'caudal_prin_m3s': caudal_estimado,
                'nivel_tc_r': np.full(horas_prediccion, nivel_inicial),
                'azud_prin': np.full(horas_prediccion, azud_inicial)
            })
            potencia_predicha = modelo.predict(X_pred)
        except Exception:
            # Curva de rendimiento de turbinas Pelton/Francis según caudal
            potencia_predicha = np.where(
                caudal_estimado >= 1.0, 
                1.82 * caudal_estimado - 0.15, 
                caudal_estimado * 0.8
            )
    else:
        # Curva de conversión hidromecánica directa para Alazán (Q max ~3.44 m3/s -> ~6.23 MW)
        potencia_predicha = np.where(
            caudal_estimado >= 1.0, 
            1.82 * caudal_estimado - 0.15, 
            caudal_estimado * 0.8
        )

    # Límite máximo de potencia permitida
    potencia_max_perm = getattr(config, 'POTENCIA_MAX_PERMITIDA_MW', 6.23)
    potencia_predicha = np.clip(potencia_predicha, 0, potencia_max_perm)

    df_pronostico['caudal_estimado'] = caudal_estimado
    df_pronostico['potencia_estimada'] = potencia_predicha

    # 5. Tarjetas de Métricas Dinámicas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Caudal Prom. Proyectado", f"{caudal_estimado.mean():.2f} m³/s")
    col2.metric("Precipitación Máx. API", f"{lluvia_api.max():.2f} mm/h")
    col3.metric("Caudal Máx. Captado", f"{caudal_estimado.max():.2f} m³/s")
    col4.metric("Potencia Máx. Proyectada", f"{potencia_predicha.max():.2f} MW")

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
            df_pronostico, 
            x='fecha_hora', 
            y=['caudal_estimado', 'precipitacion'], 
            title="Caudal Estimado (m³/s) vs Lluvia API (mm/h)",
            labels={'fecha_hora': 'Fecha y Hora', 'value': 'Magnitud', 'variable': 'Parámetro'},
            markers=True
        )
        st.plotly_chart(fig_q, use_container_width=True)

else:
    st.error("No se pudo obtener el pronóstico meteorológico de Open-Meteo. Revisa la conectividad a la API.")
