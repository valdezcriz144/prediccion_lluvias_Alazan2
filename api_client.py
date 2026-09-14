import requests
import pandas as pd
import config


def fetch_point_forecast(lat, lon):
    """Consulta la API de Open-Meteo usando el modelo de alta resolución (best_match)."""
    # Se añade 'models=best_match' para forzar la máxima resolución espacial
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=precipitation"
        f"&models=best_match"
        f"&timezone=America%2FGuayaquil"
    )

    response = requests.get(url, timeout=10)
    data = response.json()

    df = pd.DataFrame(
        {
            "fecha_hora": data["hourly"]["time"],
            "lluvia_mm": data["hourly"]["precipitation"],
        }
    )
    return df


def get_weather_forecast():
    """Obtiene el pronóstico de alta resolución para los dos puntos y aplica

    ponderación hidrológica (75% Cuenca Alta / Captación, 25% Cuenca Baja).
    """
    # 1. Consultar ambos puntos con alta resolución
    df1 = fetch_point_forecast(config.LATITUD_1, config.LONGITUD_1)  # Captación
    df2 = fetch_point_forecast(config.LATITUD_2, config.LONGITUD_2)  # Central

    # 2. Copiar estructura
    df_promedio = df1.copy()

    # 3. Ponderación hidrológica recomendada
    W_ALTA = 0.75  # 75% del peso a la captación
    W_BAJA = 0.25  # 25% del peso a la zona baja

    df_promedio["lluvia_mm"] = (df1["lluvia_mm"] * W_ALTA) + (df2["lluvia_mm"] * W_BAJA)

    return df_promedio
