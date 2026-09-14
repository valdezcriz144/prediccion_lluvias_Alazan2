import requests
import pandas as pd
import config

def fetch_point_forecast(lat, lon):
    """Consulta la API de pronóstico meteorológico para una coordenada específica."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation&timezone=America%2FGuayaquil"
    response = requests.get(url)
    data = response.json()
    
    df = pd.DataFrame({
        "fecha_hora": data["hourly"]["time"],
        "lluvia_mm": data["hourly"]["precipitation"]
    })
    return df

def get_weather_forecast():
    """Obtiene el pronóstico para los dos puntos y calcula el promedio ponderado o simple."""
    # 1. Consultar ambos puntos
    df1 = fetch_point_forecast(config.LATITUD_1, config.LONGITUD_1)
    df2 = fetch_point_forecast(config.LATITUD_2, config.LONGITUD_2)
    
    # 2. Copiar la estructura del primer dataframe
    df_promedio = df1.copy()
    
    # 3. Calcular el promedio simple (50% Punto 1 + 50% Punto 2)
    df_promedio["lluvia_mm"] = (df1["lluvia_mm"] + df2["lluvia_mm"]) / 2.0
    
    # OPCIONAL: Promedio ponderado si la cuenca alta influye más (ej. 60% Punto 1 y 40% Punto 2):
    # df_promedio["lluvia_mm"] = (df1["lluvia_mm"] * 0.6) + (df2["lluvia_mm"] * 0.4)
    
    return df_promedio
