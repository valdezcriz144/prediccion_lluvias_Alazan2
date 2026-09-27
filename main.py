from api_client import get_weather_forecast
import config
import numpy as np
import pandas as pd


def cargar_ultimo_q_turbinado_scada():
  """Lee el último Caudal Turbinado (Q_t) real registrado en la bitácora consolidada."""
  try:
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
    if col_q and not df[col_q].dropna().empty:
      return float(df[col_q].dropna().iloc[-1])
  except Exception:
    pass
  return 3.433  # Respaldo


def calcular_potencia_astec_mw(caudal_m3s):
  """Calcula la potencia ajustada a pérdidas físicas y con tope estricto de 6.23 MW."""
  q_turbinado = np.clip(caudal_m3s, 0.0, config.CAUDAL_MAX_DISEÑO)
  if q_turbinado <= 0:
    return 0.0, 0.0

  h_f = config.FACTOR_PERDIDA * (q_turbinado**2)
  h_neto = max(0.0, config.SALTO_BRUTO - h_f)
  eta_turbina = np.interp(q_turbinado, config.CURVA_Q, config.CURVA_ETA)

  potencia_watts = (
      config.GRAVEDAD
      * config.DENSIDAD_AGUA
      * q_turbinado
      * h_neto
      * eta_turbina
      * config.EFICIENCIA_GENERADOR
  )
  potencia_mw = potencia_watts / 1_000_000.0

  return min(potencia_mw, config.POTENCIA_MAX_PERMITIDA_MW), q_turbinado


def main():
  print("--- SISTEMA DE PREDICCIÓN Y GENERACIÓN HÍDRICA - ALAZÁN ---")
  print("Consultando API meteorológica y estado SCADA...")

  try:
    df_clima = get_weather_forecast()
    q_scada_actual = cargar_ultimo_q_turbinado_scada()

    lluvia_max_mm = (
        df_clima["lluvia_mm"].max() if "lluvia_mm" in df_clima.columns else 0.0
    )

    q_estimado = min(
        q_scada_actual + (lluvia_max_mm * 0.35), config.CAUDAL_MAX_DISEÑO
    )
    potencia_mw, q_turb = calcular_potencia_astec_mw(q_estimado)

    print("\nResultados con parámetros de Central Alazán:")
    print(f" -> Caudal SCADA Actual:         {q_scada_actual:.2f} m³/s")
    print(f" -> Precipitación Máx. Prevista: {lluvia_max_mm:.2f} mm/h")
    print(f" -> Caudal Estimado Proyectado:  {q_estimado:.2f} m³/s")
    print(f" -> Caudal Turbinado Efectivo:   {q_turb:.2f} m³/s")
    print(
        f" -> Potencia Generada Estimada:   {potencia_mw:.2f} MW (Máx"
        f" {config.POTENCIA_MAX_PERMITIDA_MW} MW)"
    )

    if lluvia_max_mm >= config.UMBRAL_LLUVIA_CONSIDERABLE:
      print(
          f"\n⚠️ [ALERTA LOCAL]: Lluvia considerable pronosticada"
          f" ({lluvia_max_mm:.2f} mm/h). Caudal estimado: {q_turb:.2f} m³/s |"
          f" Potencia: {potencia_mw:.2f} MW"
      )

  except Exception as e:
    print(f"[ERROR]: {e}")


if __name__ == "__main__":
  main()