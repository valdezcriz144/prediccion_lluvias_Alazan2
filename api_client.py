import config
import pandas as pd
import requests


def get_weather_forecast(lat=config.LATITUD, lon=config.LONGITUD):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=precipitation,temperature_2m,relative_humidity_2m,soil_moisture_0_to_7cm"
        f"&past_days=1"  # Permite obtener el historial de las últimas horas
        f"&forecast_days=7&timezone=auto"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error en Open-Meteo API: {response.status_code}")

    data = response.json()["hourly"]
    df = pd.DataFrame({
        "fecha_hora": pd.to_datetime(data["time"]),
        "lluvia_mm": data["precipitation"],
        "temperatura_c": data["temperature_2m"],
        "humedad_pct": data.get(
            "relative_humidity_2m", [75.0] * len(data["time"])
        ),
        "humedad_suelo": data.get(
            "soil_moisture_0_to_7cm", [0.0] * len(data["time"])
        ),
    })

    # Filtrar para conservar desde hace exactamente 3 horas en adelante
    hace_3_horas = pd.Timestamp.now().floor("H") - pd.Timedelta(hours=3)
    df = df[df["fecha_hora"] >= hace_3_horas].reset_index(drop=True)

    return df
