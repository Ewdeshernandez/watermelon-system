#!/bin/bash
# Release v3.31.1 → MAIN: Live Monitoring v2 visual refresh.
# Refactor completo de la página manteniendo backend Tier 0 A intacto.
set -e
cd "$(dirname "$0")"

VERSION="v3.31.1"

git fetch origin
git checkout dev
git pull origin dev --ff-only
python3 tests/run_smoke.py 2>&1 | tail -3

git checkout -B feat/ciclo23-2-live-visual-v2

git add pages/02_Live_Monitoring.py VERSION
git commit -m "feat(23.2): Live Monitoring v2 — visual refresh nivel internacional

  ► Mejoras visuales (vs v3.31.0)
    - 🔴 LIVE pulsante en el header (animación CSS)
    - 4 KPI cards con iconos y subtítulos: Velocidad, Sensores
      reportando, Última lectura (color por staleness), Alarmas activas
    - Tabs principales: Valores Actuales · Vectores 1X/2X · Diagnostic
      · Tendencia (en lugar de scroll vertical)
    - Status badges Normal/Alarma/Danger con colores y filas tintadas
      sutilmente — paridad visual con el legacy
    - Severidad computada desde alarm/danger del Sensor Map del activo,
      con fallback ISO 20816-3 / API 670 cuando faltan thresholds
    - Schematic embebido del activo si tiene schematic_png
    - Diagnostic con health check rápido del transducer (Gap, Bias)
    - Auto-refresh opcional cada 10s (toggle en header)
    - Tabla custom HTML con sticky headers, fonts mono para números,
      ordenamiento por severidad

  ► Sin cambios en backend
    - core/live_readings.py intacto
    - api/app.py POST /v1/ingest/live intacto
    - collector/ intacto (sigue corriendo en TES1 sin reinstalar)

  ► Por qué
    El legacy de Watermelon (watermelonsys.net/monitoreo-estatico) tenía
    mejor UI que nosotros pero menos profundidad de datos. v2 supera al
    legacy visualmente Y mantiene la ventaja de datos (vectores 1X/2X +
    diagnostic + severidad ISO).

VERSION → ${VERSION}"

git push -u origin feat/ciclo23-2-live-visual-v2
git checkout dev
git merge --no-ff feat/ciclo23-2-live-visual-v2 -m "Merge feat/ciclo23-2-live-visual-v2 into dev"
git push origin dev

# Release directo a main — solo es UI, no backend; bajo riesgo
git checkout main
git pull origin main --ff-only
git merge --no-ff dev -m "release(${VERSION}): Live Monitoring v2 visual refresh"
git tag -a "${VERSION}" -m "Release ${VERSION}: Live Monitoring v2"
git push origin main
git push origin "${VERSION}"

echo ""
echo "================================================================"
echo " ✅ ${VERSION} en MAIN — Live Monitoring v2 desplegado"
echo "================================================================"
echo ""
echo " Probar en: https://wm-home-final-2026.streamlit.app/Live_Monitoring"
echo " (esperá ~2 min al redeploy de Streamlit Cloud)"
echo ""
echo " Cambios visibles:"
echo "  • Badge 🔴 LIVE pulsante junto al título"
echo "  • 4 KPI cards arriba (incluye Alarmas activas)"
echo "  • Tabs: Valores · Vectores · Diagnostic · Tendencia"
echo "  • Status badges Normal/Alarma/Danger con colores"
echo "  • Schematic del activo si tiene"
echo "  • Auto-refresh toggle"
echo "================================================================"
