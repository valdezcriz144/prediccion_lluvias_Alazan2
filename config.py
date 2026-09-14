import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_RAW = os.path.join(BASE_DIR, "Consolidado_Sala_Control.xlsx")
FILE_CONSOLIDADO = os.path.join(BASE_DIR, "Consolidado_Filtrado.xlsx")
FILE_MODELO = os.path.join(BASE_DIR, "modelo_alazan.json")

#LATITUD = -2.517264
#LONGITUD = -78.696058

LATITUD = -2.5539
LONGITUD = -78.9286

POTENCIA_MAX_PERMITIDA_MW = 6.23
CAUDAL_MAX_DISEÑO = 3.44
CAUDAL_ECOLOGICO = 0.39  # m³/s

# Curva Empírica Oficial SCADA (Caudal vs Potencia)
TABLA_CAUDAL_SCADA = [
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.2,
    2.4,
    2.6,
    2.8,
    3.0,
    3.2,
    3.4,
    3.6,
]
TABLA_POTENCIA_SCADA = [
    1.72,
    2.07,
    2.41,
    2.75,
    3.10,
    3.44,
    3.79,
    4.13,
    4.47,
    4.82,
    5.16,
    5.51,
    5.85,
    6.20,
]

SALTO_BRUTO = 204.86
FACTOR_PERDIDA = 0.258
EFICIENCIA_GENERADOR = 0.98
GRAVEDAD = 9.81
DENSIDAD_AGUA = 1000.0

AZUD_NIVEL_MINIMO = 2.45
AZUD_NIVEL_DESBORDE = 2.83

UMBRAL_LLUVIA_CONSIDERABLE = 2.0
UMBRAL_POTENCIA_PARADA = 0.10
UMBRAL_CAUDAL_PARADA = 0.10

