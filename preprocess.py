import os
import config
import numpy as np
import pandas as pd


def cargar_y_normalizar_fechas(file_path):
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"No se encontró el archivo fuente: {file_path}")

  print(f"📥 Cargando datos desde: {file_path}")
  df = pd.read_excel(file_path)

  df.columns = [
      str(col).strip().lower().replace(" ", "_") for col in df.columns
  ]

  col_fecha_hora = next(
      (c for c in df.columns if "fecha_hora" in c or "fecha_y_hora" in c), None
  )
  col_fecha = next(
      (c for c in df.columns if "fecha" in c and "hora" not in c), None
  )
  col_hora = next(
      (c for c in df.columns if "hora" in c and "fecha" not in c), None
  )

  if col_fecha_hora:
    df["fecha_hora"] = pd.to_datetime(
        df[col_fecha_hora], dayfirst=True, errors="coerce"
    )
  elif col_fecha and col_hora:
    horas_num = (
        pd.to_numeric(df[col_hora], errors="coerce").fillna(0).astype(int)
    )
    fechas_dt = pd.to_datetime(df[col_fecha], dayfirst=True, errors="coerce")
    df["fecha_hora"] = fechas_dt + pd.to_timedelta(horas_num, unit="h")
  elif col_fecha:
    df["fecha_hora"] = pd.to_datetime(
        df[col_fecha], dayfirst=True, errors="coerce"
    )
  else:
    raise KeyError(
        "No se encontró ninguna columna de Fecha / Hora en el archivo."
    )

  df = df.dropna(subset=["fecha_hora"])
  df = df.sort_values(by="fecha_hora").drop_duplicates(subset=["fecha_hora"])
  df = df.set_index("fecha_hora")

  for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

  df_resampled = df.resample("1h").mean(numeric_only=True)
  df_interpolated = df_resampled.interpolate(
      method="linear", limit_direction="both"
  )
  df_clean = df_interpolated.ffill().bfill().reset_index()

  return df_clean


def filtrar_paradas_y_anomalias(df):
  df = df.copy()

  col_mw = next(
      (
          c
          for c in df.columns
          if any(k in c for k in ["real_mw", "potencia", "mw"])
      ),
      None,
  )
  col_q = next(
      (
          c
          for c in df.columns
          if any(
              k in c for k in ["caudal_tp_m3s", "caudal_p", "caudal", "tp"]
          )
      ),
      None,
  )

  condicion_parada = pd.Series(False, index=df.index)

  if col_mw:
    condicion_parada = condicion_parada | (
        df[col_mw].fillna(0) <= config.UMBRAL_POTENCIA_PARADA
    )
  if col_q:
    condicion_parada = condicion_parada | (
        df[col_q].fillna(0) <= config.UMBRAL_CAUDAL_PARADA
    )

  num_totales = len(df)
  num_paradas = condicion_parada.sum()
  print(
      f"⚠️ Registros filtrados por PARADA: {num_paradas} de {num_totales}"
  )

  df_operativo = df[~condicion_parada].copy().reset_index(drop=True)
  return df_operativo, col_q, col_mw


def calcular_potencia_fisica_alazan(df, col_q):
  """Calcula la potencia hidráulica usando la curva oficial SCADA de la central."""
  df = df.copy()

  if col_q and col_q in df.columns:
    q_clamped = np.clip(df[col_q], 0.0, config.CAUDAL_MAX_DISEÑO)

    # Interpolar potencia directamente desde la tabla SCADA
    potencia_mw = np.interp(
        q_clamped, config.TABLA_CAUDAL_SCADA, config.TABLA_POTENCIA_SCADA
    )

    df["potencia_teorica_mw"] = np.clip(
        potencia_mw, 0.0, config.POTENCIA_MAX_PERMITIDA_MW
    )

  return df


def generar_features_ml(df, col_q):
  df = df.copy()

  df["hora"] = df["fecha_hora"].dt.hour
  df["mes"] = df["fecha_hora"].dt.month
  df["dia_semana"] = df["fecha_hora"].dt.dayofweek

  df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24.0)
  df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24.0)
  df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12.0)
  df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12.0)

  if col_q and col_q in df.columns:
    for lag in [1, 2, 3, 6, 12, 24]:
      df[f"{col_q}_lag_{lag}h"] = df[col_q].shift(lag)

    df[f"{col_q}_roll_mean_3h"] = (
        df[col_q].rolling(window=3, min_periods=1).mean()
    )
    df[f"{col_q}_roll_mean_6h"] = (
        df[col_q].rolling(window=6, min_periods=1).mean()
    )
    df[f"{col_q}_roll_mean_24h"] = (
        df[col_q].rolling(window=24, min_periods=1).mean()
    )

  df = df.ffill().bfill()
  return df


def ejecutar_pipeline_procesamiento():
  print("=" * 60)
  print("🚀 PIPELINE DE LIMPIEZA CON CURVA SCADA OFICIAL")
  print("=" * 60)

  df_raw = cargar_y_normalizar_fechas(config.FILE_RAW)
  df_operativo, col_q, col_mw = filtrar_paradas_y_anomalias(df_raw)
  df_fisica = calcular_potencia_fisica_alazan(df_operativo, col_q)
  df_final = generar_features_ml(df_fisica, col_q)

  df_final.to_excel(config.FILE_CONSOLIDADO, index=False)

  print("\n✅ Procesamiento completado correctamente.")
  print(
      f"📊 Filas procesadas: {len(df_final):,} | Variables generadas:"
      f" {len(df_final.columns)}"
  )
  print("=" * 60)

  return df_final


if __name__ == "__main__":
  ejecutar_pipeline_procesamiento()