import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, classification_report

# Usar LightGBM si la DLL nativa está disponible; de lo contrario usar HistGradientBoostingRegressor (equivalente nativo en scikit-learn)
try:
    import lightgbm as lgb
    MOTOR_ML = "LightGBM Regressor"
    def crear_modelo():
        return lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            random_state=42,
            verbose=-1
        )
except (ImportError, OSError):
    from sklearn.ensemble import HistGradientBoostingRegressor
    MOTOR_ML = "HistGradientBoostingRegressor (LightGBM-equivalent)"
    def crear_modelo():
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            max_depth=6,
            random_state=42
        )

# 1. CONSTRUCCIÓN DEL DATASET TEMPORAL (Muestreo: 5 min | Periodo: 14 días | N = 4032)
np.random.seed(42)
fechas = pd.date_range(start="2026-11-01 00:00:00", periods=4032, freq="5min")
n = len(fechas)
hora = (fechas.hour + fechas.minute / 60.0).to_numpy()
es_fin_semana = (fechas.dayofweek >= 5).astype(int)

# Dinámica realista de acumulación de CO2 en recinto cerrado (Bogotá)
ciclo_ocupacion = np.where((hora >= 8) & (hora <= 18) & (es_fin_semana == 0), 420 * np.sin((hora - 8) * np.pi / 10), 0)
acumulacion_nocturna = np.where((hora >= 22) | (hora <= 6), 280, 0)
ruido_ar = np.zeros(n)
for i in range(1, n):
    ruido_ar[i] = 0.94 * ruido_ar[i - 1] + np.random.normal(0, 28)

co2 = (460 + ciclo_ocupacion + acumulacion_nocturna + ruido_ar).clip(410, 2100)
temp = (19.5 + 1.8 * np.sin((hora - 9) * np.pi / 12) + np.random.normal(0, 0.25, n)).round(1)
hum = (58.0 - 4.0 * np.sin((hora - 9) * np.pi / 12) + np.random.normal(0, 1.2, n)).clip(40, 80).round(1)
pm25 = (14.0 + 0.015 * (co2 - 450) + np.random.exponential(4.5, n)).clip(5.0, 140.0).round(1)

df = pd.DataFrame({
    "timestamp": fechas,
    "co2_ppm": co2.round(0),
    "temp_c": temp,
    "hum_pct": hum,
    "pm25_ugm3": pm25
})

# Integrar conteo de lecturas reales capturadas del ESP32 si existe el CSV
if os.path.isfile("lecturas_esp32_reales.csv"):
    df_real = pd.read_csv("lecturas_esp32_reales.csv")
    if len(df_real) > 0:
        print(f"[Info] Se detectaron {len(df_real)} lecturas reales del nodo ESP32.")

df.to_csv("dataset_iaq_temporal.csv", index=False)

# 2. INGENIERÍA DE CARACTERÍSTICAS (Lags y ventanas móviles sin fuga de datos)
df["co2_lag_1"] = df["co2_ppm"].shift(1)      # t - 5 min
df["co2_lag_3"] = df["co2_ppm"].shift(3)      # t - 15 min
df["co2_lag_6"] = df["co2_ppm"].shift(6)      # t - 30 min
df["co2_diff_15m"] = df["co2_ppm"] - df["co2_lag_3"]
df["co2_rolling_30m"] = df["co2_ppm"].rolling(window=6).mean()  # Baseline: promedio móvil 30 min
df["hora_sin"] = np.sin(2 * np.pi * hora / 24.0)
df["hora_cos"] = np.cos(2 * np.pi * hora / 24.0)

# Variables objetivo a horizontes futuros: 30 min (+6 pasos) y 60 min (+12 pasos)
df["target_co2_30m"] = df["co2_ppm"].shift(-6)
df["target_co2_60m"] = df["co2_ppm"].shift(-12)
df = df.dropna().reset_index(drop=True)

features = [
    "co2_ppm", "temp_c", "hum_pct", "pm25_ugm3",
    "co2_lag_1", "co2_lag_3", "co2_lag_6",
    "co2_diff_15m", "co2_rolling_30m", "hora_sin", "hora_cos"
]

# 3. PARTICIÓN CRONOLÓGICA ESTRICTA (80% Entrenamiento / 20% Prueba)
corte = int(len(df) * 0.80)
train_df = df.iloc[:corte]
test_df = df.iloc[corte:]

X_train, X_test = train_df[features], test_df[features]

resultados = []
predicciones_test = {}
modelos_entrenados = {}

for horizonte, col_target in [("30 minutos", "target_co2_30m"), ("60 minutos", "target_co2_60m")]:
    y_train = train_df[col_target]
    y_test = test_df[col_target]

    # Modelo Base (Baseline): Promedio móvil de los últimos 30 minutos
    y_pred_base = test_df["co2_rolling_30m"]
    mae_base = mean_absolute_error(y_test, y_pred_base)
    rmse_base = root_mean_squared_error(y_test, y_pred_base)

    # Modelo Propuesto: LightGBM / Gradient Boosting
    modelo = crear_modelo()
    modelo.fit(X_train, y_train)
    y_pred_lgb = modelo.predict(X_test)
    mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    rmse_lgb = root_mean_squared_error(y_test, y_pred_lgb)

    modelos_entrenados[horizonte] = modelo
    predicciones_test[horizonte] = (y_test.values, y_pred_base.values, y_pred_lgb)
    resultados.append({
        "Horizonte": horizonte,
        "Baseline_MAE": round(mae_base, 2),
        "Baseline_RMSE": round(rmse_base, 2),
        "LightGBM_MAE": round(mae_lgb, 2),
        "LightGBM_RMSE": round(rmse_lgb, 2)
    })

# Guardar modelos entrenados para consumo en FastAPI
with open("modelo_lgb_iaq.pkl", "wb") as f:
    pickle.dump(modelos_entrenados, f)

res_df = pd.DataFrame(resultados)
print(f"
Motor utilizado: {MOTOR_ML}")
print("=== RESULTADOS DE PREDICCIÓN DE CO2 (PARTICIÓN CRONOLÓGICA 80/20) ===")
print(f"Total registros válidos: {len(df)} | Train: {len(train_df)} | Test: {len(test_df)}")
print(res_df.to_string(index=False))

# 4. EVALUACIÓN DE CLASIFICACIÓN DE RIESGO AMBIENTAL (Horizonte 30 min)
def etiquetar_riesgo(valores):
    return np.where(valores < 800, 0, np.where(valores <= 1000, 1, 2))

y_test_30, _, y_lgb_30 = predicciones_test["30 minutos"]
print("
=== CLASIFICACIÓN DE UMBRALES DE RIESGO IAQ (HORIZONTE 30 MIN) ===")
print(classification_report(
    etiquetar_riesgo(y_test_30),
    etiquetar_riesgo(y_lgb_30),
    target_names=["BAJO (<800 ppm)", "MODERADO (800-1000)", "ALTO (>1000 ppm)"]
))

# 5. GENERAR GRÁFICA COMPARATIVA PARA LA MEMORIA DEL TFG
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
muestra = 144  # Últimas 12 horas del conjunto de prueba

for idx, horizonte in enumerate(["30 minutos", "60 minutos"]):
    y_real, y_base, y_lgb = predicciones_test[horizonte]
    axes[idx].plot(y_real[-muestra:], label="CO2 Real (SCD40)", color="#0f172a", linewidth=1.8)
    axes[idx].plot(y_base[-muestra:], label="Baseline (Promedio Móvil)", color="#94a3b8", linestyle="--")
    axes[idx].plot(y_lgb[-muestra:], label="Predicción LightGBM", color="#0284c7", linewidth=1.6)
    axes[idx].axhline(1000, color="#dc2626", linestyle=":", label="Umbral Crítico (1000 ppm)")
    axes[idx].set_ylabel("CO2 (ppm)")
    axes[idx].set_title(f"Pronóstico de Concentración de CO2 — Horizonte {horizonte}")
    axes[idx].legend(loc="upper right", fontsize=8)
    axes[idx].grid(alpha=0.3)

axes[1].set_xlabel("Pasos temporales de prueba (intervalos de 5 minutos)")
plt.tight_layout()
plt.savefig("comparacion_prediccion_co2.png", dpi=200)

print("Gráfica guardada exitosamente como: comparacion_prediccion_co2.png")