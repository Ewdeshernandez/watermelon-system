#!/bin/bash
# Release v3.31.2 → MAIN: Live Monitoring v3 industrial pro.
# Hero machine map con sensor dots + sparklines + phasors. Sin marcas externas.
set -e
cd "$(dirname "$0")"

VERSION="v3.31.2"

git fetch origin

# 1. Validar dev al día (puede o no estar al día con main)
git checkout dev
git pull origin dev --ff-only
python3 tests/run_smoke.py 2>&1 | tail -3

# 2. Mover los cambios actuales (v3) al flujo dev → main
#    Como editamos en main directamente la última vez, ahora hacemos
#    lo mismo: commit on main, then sync dev.

git checkout main
git pull origin main --ff-only

# Si hay cambios sin commit (nuestros archivos v3), los agregamos
git add core/live_readings.py pages/02_Live_Monitoring.py VERSION

# Crear commit (puede fallar si no hay cambios — usamos --allow-empty para idempotencia)
git commit -m "feat(23.3): Live Monitoring v3 — pro industrial control room

  ► Cambios visuales mayores
    - Hero Live Sensor Map: schematic con sensor dots vivos
      (verde/ámbar/rojo, pulse en danger), tooltip por hover.
    - Sparklines SVG en cada fila de canales — micro trend de
      últimas 30 lecturas inline, sin penalizar latencia (1 query
      grouped en core/live_readings.recent_history_all_direct).
    - Phasor mini-charts del 1X (vector amplitud + fase) por
      sensor. Polar plot con escala consistente y filtrado de
      fases inválidas (~1e-41 cuando ampl es ~0).
    - Alarm strip prominente cuando hay sensores en danger
      (banner rojo arriba con los críticos).
    - Trend chart con bandas de severidad (Plotly) — verde/ámbar/
      rojo de fondo según thresholds del sensor o ISO fallback.
    - Tabla con headers oscuros, filas tintadas por estado,
      tabular-nums para valores, monospace para units.
    - Header limpio: NO menciona vendor de hardware. Subtitle
      generic pro: 'Real-time machine health · ISO 20816 / API 670'.

  ► Backend
    - core/live_readings.recent_history_all_direct: una query
      agrupada para todos los sparklines de un activo.
    - api / collector / supabase intactos.

  ► Por qué
    Feedback usuario: 'se ve muy pobre, muy de PowerPoint, mejor
    que System1/Emerson/SKF y mejor que el legacy de Watermelon'.

VERSION → ${VERSION}" --allow-empty

# 3. Tag + push main
git tag -a "${VERSION}" -m "Release ${VERSION}: Live Monitoring v3 pro industrial"
git push origin main
git push origin "${VERSION}"

# 4. Sync dev con main (no queremos que dev quede atrás)
git checkout dev
git pull origin dev --ff-only
git merge main --no-ff -m "Sync dev with main ${VERSION}"
git push origin dev

git checkout main

echo ""
echo "================================================================"
echo " ✅ ${VERSION} en MAIN — Live Monitoring v3 pro industrial"
echo "================================================================"
echo ""
echo " Refrescá: https://wm-home-final-2026.streamlit.app/Live_Monitoring"
echo " Esperá ~2 min al redeploy. Cmd+Shift+R para evitar cache."
echo ""
echo " Verás:"
echo "  • Hero machine map con sensor dots vivos"
echo "  • Sparklines en cada canal"
echo "  • Phasor mini-charts en Vectores 1X/2X"
echo "  • Alarm strip rojo si hay danger"
echo "  • Trend chart con bandas de severidad"
echo "================================================================"
