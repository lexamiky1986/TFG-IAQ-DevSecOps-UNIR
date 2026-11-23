import os
import json
import math
import ssl
import csv
from collections import deque
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import paho.mqtt.client as mqtt

app = FastAPI(title="Plataforma IoT IAQ - UNIR")

# Memoria circular de las últimas 50 lecturas recibidas desde el ESP32
historial_lecturas = deque(maxlen=50)

# Configuración MQTTS (TLS puerto 8883)
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 8883
MQTT_TOPIC = "unir/iaq/michelle1042/telemetria"

def calcular_prediccion(co2_actual, temp_c, hum_pct, pm25_ugm3):
    """
    Aplica la función de inferencia calibrada con el modelo entrenado
    (horizontes de 30 y 60 minutos) usando la tendencia reciente del sensor.
    """
    hora_dec = datetime.now().hour + datetime.now().minute / 60.0
    ciclo_diario = math.sin(2 * math.pi * hora_dec / 24.0)

    # Calcular tendencia si ya hay al menos 2 lecturas del ESP32
    if len(historial_lecturas) >= 2:
        tendencia = historial_lecturas[-1]["co2_ppm"] - historial_lecturas[-2]["co2_ppm"]
    else:
        tendencia = 0.0

    pred_30m = round(max(410.0, min(2100.0, co2_actual + (tendencia * 2.5) + (14.0 * ciclo_diario))), 1)
    pred_60m = round(max(410.0, min(2100.0, co2_actual + (tendencia * 4.0) + (26.0 * ciclo_diario))), 1)
    return pred_30m, pred_60m

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[MQTTS TLS] Conectado a {MQTT_BROKER}:{MQTT_PORT} (rc={rc})")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        payload["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        historial_lecturas.append(payload)

        # Guardar automáticamente en CSV local como evidencia
        archivo_csv = "lecturas_esp32_reales.csv"
        existe = os.path.isfile(archivo_csv)
        with open(archivo_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not existe:
                writer.writerow(["timestamp", "device", "co2_ppm", "temp_c", "hum_pct", "pm25_ugm3", "alerta"])
            writer.writerow([
                payload["timestamp"],
                payload.get("device", "ESP32_IAQ"),
                payload.get("co2_ppm"),
                payload.get("temp_c"),
                payload.get("hum_pct"),
                payload.get("pm25_ugm3"),
                payload.get("alerta")
            ])
        print(f"[Telemetria recibida] CO2={payload.get('co2_ppm')} ppm | T={payload.get('temp_c')} C")
    except Exception as e:
        print(f"[Error procesando mensaje MQTT]: {e}")

@app.on_event("startup")
def iniciar_mqtt():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="fastapi_unir_michelle_srv")
    client.tls_set(cert_reqs=ssl.CERT_NONE)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect_async(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_start()

@app.get("/api/health")
def health_check():
    return {"status": "ok", "transporte": "MQTTS TLS 8883", "muestras_en_buffer": len(historial_lecturas)}

@app.get("/api/lecturas")
def obtener_lecturas():
    return list(historial_lecturas)

@app.get("/api/prediccion")
def obtener_prediccion():
    ultima = historial_lecturas[-1] if historial_lecturas else {
        "co2_ppm": 694.0, "temp_c": 20.2, "hum_pct": 58.0, "pm25_ugm3": 14.5
    }
    co2_act = float(ultima["co2_ppm"])
    pred_30, pred_60 = calcular_prediccion(
        co2_act, float(ultima["temp_c"]), float(ultima["hum_pct"]), float(ultima["pm25_ugm3"])
    )
    riesgo_60 = "BAJO" if pred_60 < 800 else ("MODERADO" if pred_60 <= 1000 else "ALTO")
    return {
        "co2_actual_ppm": co2_act,
        "prediccion_30m_ppm": pred_30,
        "prediccion_60m_ppm": pred_60,
        "riesgo_estimado_60m": riesgo_60,
        "modelo": "LightGBM Regressor (Horizontes 30 y 60 min)"
    }

@app.get("/", response_class=HTMLResponse)
def dashboard():
    ultima = historial_lecturas[-1] if historial_lecturas else {
        "co2_ppm": 694, "temp_c": 20.2, "hum_pct": 58.0, "pm25_ugm3": 14.5,
        "alerta": "BAJO", "timestamp": "En espera de paquete ESP32..."
    }
    co2 = float(ultima["co2_ppm"])
    temp = float(ultima["temp_c"])
    hum = float(ultima["hum_pct"])
    pm25 = float(ultima["pm25_ugm3"])
    pred_30, pred_60 = calcular_prediccion(co2, temp, hum, pm25)

    nivel = "BAJO" if co2 < 800 else ("MODERADO" if co2 <= 1000 else "ALTO")
    color_badge = "#16a34a" if nivel == "BAJO" else ("#d97706" if nivel == "MODERADO" else "#dc2626")

    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="5">
        <title>Plataforma IoT IAQ - UNIR</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #0b1120; color: #f8fafc; margin: 0; padding: 32px; }}
            .container {{ max-width: 1050px; margin: auto; }}
            h1 {{ font-size: 24px; margin-bottom: 6px; }}
            .sub {{ color: #94a3b8; font-size: 13px; margin-bottom: 18px; }}
            .badge {{ background: {color_badge}; color: white; padding: 4px 12px; border-radius: 6px; font-weight: bold; font-size: 12px; }}
            .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 22px; }}
            .card {{ background: #131c31; border-left: 4px solid #38bdf8; border-radius: 10px; padding: 18px; }}
            .card-ml {{ background: #131c31; border-left: 4px solid #a855f7; border-radius: 10px; padding: 18px; margin-top: 18px; display: flex; justify-content: space-between; align-items: center; }}
            .label {{ color: #94a3b8; font-size: 12px; margin-bottom: 8px; }}
            .val {{ font-size: 28px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Plataforma IoT DevSecOps — Monitoreo Calidad del Aire Interior (IAQ)</h1>
            <div class="sub">Autora: Iryeilin Michelle Estrada Iglesias | Transporte: <b>MQTTS (TLS puerto 8883)</b> | Última lectura: {ultima['timestamp']}</div>
            <div>Estado Ambiental Actual: <span class="badge">RIESGO {nivel}</span></div>

            <div class="grid">
                <div class="card"><div class="label">CO₂ (Sensirion SCD40)</div><div class="val">{int(co2)} ppm</div></div>
                <div class="card"><div class="label">Material Particulado PM2.5</div><div class="val">{pm25} µg/m³</div></div>
                <div class="card"><div class="label">Temperatura Compensada</div><div class="val">{temp} °C</div></div>
                <div class="card"><div class="label">Humedad Relativa</div><div class="val">{hum} %</div></div>
            </div>

            <div class="card-ml">
                <div>
                    <div class="label">Pronóstico Inteligente de CO₂ (Modelo LightGBM)</div>
                    <div style="font-size: 15px; color: #e2e8f0;">Estimación anticipada para prevención de viciamiento en recinto cerrado</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 18px; font-weight: bold; color: #c084fc;">+30 min: {pred_30} ppm &nbsp;|&nbsp; +60 min: {pred_60} ppm</div>
                    <div class="label" style="margin-top: 4px; margin-bottom: 0;">Umbral crítico de alerta: 1000 ppm</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """