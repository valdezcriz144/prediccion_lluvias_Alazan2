import os
import config
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb


def q_a_potencia_mw(q_val):
  """Convierte Caudal a Potencia (MW) usando la curva oficial SCADA."""
  q_clamped = np.clip(q_val, 0.0, config.CAUDAL_MAX_DISEÑO)
  pot_mw = np.interp(
      q_clamped, config.TABLA_CAUDAL_SCADA, config.TABLA_POTENCIA_SCADA
  )
  return np.clip(pot_mw, 0.0, config.POTENCIA_MAX_PERMITIDA_MW)


def entrenar():
  print("=" * 55)
  print("🚀 ENTRENANDO XGBOOST CON CURVA OFICIAL SCADA")
  print("=" * 55)

  if not os.path.exists(config.FILE_CONSOLIDADO):
    print(f"❌ [ERROR]: Ejecuta preprocess.py primero.")
    return

  df = pd.read_excel(config.FILE_CONSOLIDADO)
  df.columns = [str(c).strip() for c in df.columns]

  col_fecha = next(
      (c for c in df.columns if c.lower() in ["fecha_hora", "fecha", "time"]),
      None,
  )
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

  if not col_q:
    print("❌ [ERROR]: No se detectó la columna de caudal.")
    return

  if col_fecha:
    df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce")
    df = df.dropna(subset=[col_fecha]).sort_values(by=col_fecha)

  df["Q_turb"] = (
      pd.to_numeric(df[col_q], errors="coerce")
      .fillna(0.0)
      .clip(0, config.CAUDAL_MAX_DISEÑO)
  )

  col_lag1 = f"{col_q}_lag_1h"
  col_lag2 = f"{col_q}_lag_2h"

  df["Q_lag1"] = (
      df[col_lag1]
      if col_lag1 in df.columns
      else df["Q_turb"].shift(1).bfill()
  )
  df["Q_lag2"] = (
      df[col_lag2]
      if col_lag2 in df.columns
      else df["Q_turb"].shift(2).bfill()
  )

  if col_fecha:
    df["mes"] = df[col_fecha].dt.month
    df["hora"] = df[col_fecha].dt.hour
    df["dia_semana"] = df[col_fecha].dt.dayofweek
  else:
    df["mes"], df["hora"], df["dia_semana"] = 8, 12, 0

  feature_cols = ["Q_lag1", "Q_lag2", "mes", "hora", "dia_semana"]
  X = df[feature_cols]
  y = df["Q_turb"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, shuffle=False
  )

  model = xgb.XGBRegressor(
      n_estimators=100,
      learning_rate=0.05,
      max_depth=5,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42,
  )
  model.fit(X_train, y_train)

  q_preds = model.predict(X_test)
  p_real_mw = q_a_potencia_mw(y_test.values)
  p_pred_mw = q_a_potencia_mw(q_preds)

  mae = mean_absolute_error(p_real_mw, p_pred_mw)
  rmse = np.sqrt(mean_squared_error(p_real_mw, p_pred_mw))

  print("\n" + "=" * 55)
  print("📊 RESULTADOS DEL MODELO AJUSTADO A SCADA")
  print("=" * 55)
  print(f"• MAE  : {mae:.3f} MW")
  print(f"• RMSE : {rmse:.3f} MW")

  model.save_model(config.FILE_MODELO)
  print(f"\n💾 Modelo guardado exitosamente en '{config.FILE_MODELO}'.")


if __name__ == "__main__":
  entrenar()