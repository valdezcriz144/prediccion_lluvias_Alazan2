import requests
import pandas as pd
import config


def fetch_point_forecast(lat, lon):
    """Consulta la API de Open-Meteo usando el modelo de alta resolución (best_match)."""
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
    """Obtiene el pronóstico de alta resolución para 3 puntos de la cuenca

    y aplica ponderación hidrológica (Alta: 50%, Media: 35%, Baja: 15%).
    """
    # 1. Consultar los 3 puntos
    df1 = fetch_point_forecast(config.LATITUD_1, config.LONGITUD_1)  # Alta / Captación
    df_media = fetch_point_forecast(config.LATITUD_3, config.LONGITUD_3)  # Media / Intermedio
    df2 = fetch_point_forecast(config.LATITUD_2, config.LONGITUD_2)  # Baja / Central

    # 2. Copiar estructura
    df_promedio = df1.copy()

    # 3. Ponderación hidrológica de los 3 puntos (Suma = 1.0)
    W_ALTA = 0.50   # 50% peso cuenca alta
    W_MEDIA = 0.35  # 35% peso cuenca media
    W_BAJA = 0.15   # 15% peso cuenca baja

    df_promedio["lluvia_mm"] = (
        (df1["lluvia_mm"] * W_ALTA) +
        (df_media["lluvia_mm"] * W_MEDIA) +
        (df2["lluvia_mm"] * W_BAJA)
    )

    return df_promedio
