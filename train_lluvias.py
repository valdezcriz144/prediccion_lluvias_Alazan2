import pandas as pd
from xgboost import XGBRegressor
import warnings

# Ignorar advertencias de openpyxl
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

def procesar_lluvias(filepath="Estación pluviométrica_Alazán.xlsx"):
    print("Procesando datos de lluvia...")
    # Leer la hoja 'Datos'
    df_lluvia = pd.read_excel(filepath, sheet_name="Datos")
    
    # Crear índice temporal a partir de las columnas de fecha y hora
    df_lluvia['Timestamp'] = pd.to_datetime(
        df_lluvia['Fecha (dd/mm/aa)'].dt.strftime('%Y-%m-%d') + ' ' + 
        df_lluvia['Hora (hh:mm:ss)'].astype(str)
    )
    df_lluvia = df_lluvia.sort_values('Timestamp').set_index('Timestamp')
    
    # Extraer y limpiar precipitación
    df_precip = df_lluvia[['Precipitación (mm)']].copy()
    df_precip['Precipitación (mm)'] = df_precip['Precipitación (mm)'].fillna(0)
    
    # Remuestrear a formato horario (si hay múltiples registros por hora, promedia; si faltan, rellena con 0)
    df_precip = df_precip.resample('H').mean().fillna(0)
    
    # 1. Crear variables de retardo (Time Lags) para simular el tiempo de viaje del agua
    for lag in [1, 2, 3, 6, 12]:
        df_precip[f'precip_lag_{lag}h'] = df_precip['Precipitación (mm)'].shift(lag)
        
    # 2. Crear variables de acumulados para simular saturación de suelo (24h y 72h)
    df_precip['precip_sum_24h'] = df_precip['Precipitación (mm)'].rolling(window=24).sum()
    df_precip['precip_sum_72h'] = df_precip['Precipitación (mm)'].rolling(window=72).sum()
    
    # Eliminar las filas iniciales que quedan vacías (NaN) por el cálculo de lags y rolling
    return df_precip.dropna()


def cargar_datos_operativos(filepath="Consolidado_Filtrado.xlsx"):
    print("Cargando datos operativos...")
    df_ope = pd.read_excel(filepath)
    
    # IMPORTANTE: Reemplaza 'Columna_Fecha' por el nombre exacto de la columna de fecha/hora en tu Excel consolidado
    df_ope['Timestamp'] = pd.to_datetime(df_ope['Columna_Fecha']) 
    df_ope = df_ope.set_index('Timestamp')
    
    return df_ope


def crear_modelo():
    df_clima = procesar_lluvias("Estación pluviométrica_Alazán.xlsx")
    df_operativo = cargar_datos_operativos("Consolidado_Filtrado.xlsx")
    
    print("Fusionando datasets...")
    # Une ambos archivos asegurando que la fecha y hora coincidan exactamente
    df_final = df_operativo.join(df_clima, how='inner').dropna()
    
    if df_final.empty:
        print("Error: No hay fechas que coincidan entre el archivo consolidado y el pluviométrico.")
        return

    # IMPORTANTE: Ajusta estas variables para que coincidan con las columnas de Consolidado_Filtrado.xlsx
    features = [
        'Caudal_Turbinado', # Reemplaza con tu nombre de columna real
        'Nivel_Embalse',    # Reemplaza con tu nombre de columna real
        'precip_lag_1h',
        'precip_lag_3h',
        'precip_lag_6h',
        'precip_sum_24h',
        'precip_sum_72h'
    ]
    target = 'Potencia_Generada' # Reemplaza con tu columna objetivo real
    
    X = df_final[features]
    y = df_final[target]
    
    print("Entrenando modelo XGBoost...")
    modelo = XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    modelo.fit(X, y)
    
    print("Guardando modelo actualizado...")
    modelo.save_model("modelo_alazan.json")
    print("¡Éxito! El modelo se ha guardado como 'modelo_alazan.json'.")


if __name__ == "__main__":
    crear_modelo()
