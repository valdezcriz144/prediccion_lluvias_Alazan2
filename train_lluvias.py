import pandas as pd
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

def procesar_lluvias(filepath="Estación pluviométrica_Alazán.xlsx"):
    print("Procesando datos de lluvia...")
    df_lluvia = pd.read_excel(filepath, sheet_name="Datos")
    
    # Crear índice temporal combinando fecha y hora
    df_lluvia['Timestamp'] = pd.to_datetime(
        df_lluvia['Fecha (dd/mm/aa)'].dt.strftime('%Y-%m-%d') + ' ' + 
        df_lluvia['Hora (hh:mm:ss)'].astype(str)
    )
    df_lluvia = df_lluvia.sort_values('Timestamp').set_index('Timestamp')
    
    # Extraer y limpiar precipitación
    df_precip = df_lluvia[['Precipitación (mm)']].copy()
    df_precip['Precipitación (mm)'] = df_precip['Precipitación (mm)'].fillna(0)
    
    # Normalizar a frecuencia horaria
    df_precip = df_precip.resample('H').mean().fillna(0)
    
    # Retardos (Lags) para el tiempo de concentración de la cuenca
    for lag in [1, 2, 3, 6, 12]:
        df_precip[f'precip_lag_{lag}h'] = df_precip['Precipitación (mm)'].shift(lag)
        
    # Acumulados móviles para la saturación del suelo (24h y 72h)
    df_precip['precip_sum_24h'] = df_precip['Precipitación (mm)'].rolling(window=24).sum()
    df_precip['precip_sum_72h'] = df_precip['Precipitación (mm)'].rolling(window=72).sum()
    
    return df_precip.dropna()


def cargar_datos_operativos(filepath="Consolidado_Filtrado.xlsx"):
    print("Cargando datos operativos...")
    df_ope = pd.read_excel(filepath)
    
    # Convertir columna 'fecha_hor' a datetime y definir como índice
    df_ope['Timestamp'] = pd.to_datetime(df_ope['fecha_hor']) 
    df_ope = df_ope.set_index('Timestamp')
    
    return df_ope


def crear_modelo():
    df_clima = procesar_lluvias("Estación pluviométrica_Alazán.xlsx")
    df_operativo = cargar_datos_operativos("Consolidado_Filtrado.xlsx")
    
    print("Fusionando datasets por timestamp...")
    df_final = df_operativo.join(df_clima, how='inner').dropna()
    
    if df_final.empty:
        print("Error: No coinciden los rangos de fechas entre ambos archivos Excel.")
        return

    # Selección de variables según las columnas de tus archivos
    features = [
        'caudal_prin_m3s',
        'nivel_tc_r',
        'azud_prin',
        'precip_lag_1h',
        'precip_lag_3h',
        'precip_lag_6h',
        'precip_sum_24h',
        'precip_sum_72h'
    ]
    target = 'real_mw'
    
    X = df_final[features]
    y = df_final[target]
    
    print(f"Entrenando modelo XGBoost con {len(df_final)} registros coincidentes...")
    modelo = XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    modelo.fit(X, y)
    
    print("Guardando modelo actualizado...")
    modelo.save_model("modelo_alazan.json")
    print("¡Éxito! El archivo 'modelo_alazan.json' ha sido generado con las variables pluviométricas.")


if __name__ == "__main__":
    crear_modelo()
