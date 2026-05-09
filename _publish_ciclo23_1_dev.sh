#!/bin/bash
# Ciclo 23.1 → DEV: Tier 0 A — Live Data Ingestion (Modbus → API → Supabase).
# Backend completo: collector Windows + endpoint + tabla Supabase + página Streamlit.
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B feat/ciclo23-1-live-ingestion

git add \
  data/modbus_maps/tes1.json \
  supabase/migrations/20260508_001_live_readings.sql \
  core/live_readings.py \
  api/app.py \
  collector/wm_collector.py \
  collector/wm_collector.config.example.json \
  collector/requirements.txt \
  collector/install_windows.ps1 \
  collector/README.md \
  pages/02_Live_Monitoring.py \
  VERSION

git commit -m "feat(23.1): Tier 0 A — Live Data Ingestion (Modbus → API → Supabase)

  ► Pipeline completo
    PC planta + ZeroTier ──Modbus TCP──> Bently 3500/92
    PC planta ────────────HTTPS POST───> wm-api Render
    wm-api ───────────────INSERT───────> Supabase live_readings
    Streamlit ────────────SELECT───────> render dashboard

  ► Componentes nuevos
    - data/modbus_maps/tes1.json — mapa completo TES1 (8 sensores +
      vectores 1X/2X + Gap + BiasVoltage + Velocidad).
    - supabase/migrations/20260508_001_live_readings.sql — tabla
      append-only + view latest_live_reading + RLS.
    - core/live_readings.py — capa de persistencia (LiveReading,
      ingest_batch, latest_for_instance, history_for_metric).
    - api/app.py — endpoint POST /v1/ingest/live con Pydantic models
      (LiveIngestPayload, LiveReadingItem). Auth Bearer.
    - collector/wm_collector.py — script Python liviano (~400 LOC)
      que corre en PC de planta. Lee Modbus cada 10s, decodifica
      float32 (4 byte orders), POST con buffer SQLite resiliente.
    - collector/install_windows.ps1 — instalador NSSM Windows
      (auto-start, auto-restart, logs rotativos).
    - pages/02_Live_Monitoring.py — UI: health strip + valores
      actuales + vectores 1X/2X + diagnostic + tendencia histórica.

  ► Diferenciador estratégico
    - System1/AMS Suite cobran ~80k USD/año por feature equivalente.
    - Nosotros entregamos vectores 1X/2X (Ampl + Phase) gratis —
      mismo dato que el 3500/92 ya envía via Modbus.

  ► Próximos pasos para activar TES1
    1. Aplicar migración SQL en Supabase Dashboard (SQL Editor).
    2. Verificar que watermelon-api en Render redeployó con el
       endpoint nuevo (revisar /docs).
    3. RDP al PC de planta TES1 (192.168.192.108).
    4. Correr install_windows.ps1 con la api_key del collector.
    5. Comparar valores en pages/02_Live_Monitoring contra
       https://watermelonsys.net/monitoreo-estatico (ground truth).

VERSION → v3.31.0-dev"

git push -u origin feat/ciclo23-1-live-ingestion
git checkout dev
git merge --no-ff feat/ciclo23-1-live-ingestion -m "Merge feat/ciclo23-1-live-ingestion into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 23.1 en DEV — v3.31.0-dev"
echo " Tier 0 A: Live Ingestion pipeline completo."
echo ""
echo " PARA PROBAR:"
echo "  1. Supabase SQL Editor → correr"
echo "     supabase/migrations/20260508_001_live_readings.sql"
echo "  2. Verificar Render redeployó (debería ya tener el endpoint)"
echo "     curl https://watermelon-api-bpv4.onrender.com/v1/health"
echo "  3. RDP al PC TES1 (192.168.192.108)"
echo "  4. Copiar carpeta collector/ + tes1.json"
echo "  5. PowerShell admin → ./install_windows.ps1 con tus paths"
echo "  6. Abrir wm-test → Live Monitoring → seleccionar tes1"
echo "  7. Comparar contra watermelonsys.net/monitoreo-estatico"
echo "================================================================"
