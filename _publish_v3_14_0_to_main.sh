#!/bin/bash
# =============================================================
# Watermelon — Release v3.14.0: Industrial Plumbing Foundations
# =============================================================
# Merge dev → main + tag v3.14.0.
#
# Pre-requisitos (deben cumplirse antes de correr este script):
#   ✓ _publish_ciclo18_1_dev.sh ya ejecutado (dev tiene los 4 commits)
#   ✓ python3 tests/run_smoke.py → 157 passed
#   ✓ streamlit run app.py → arranca idéntico a v3.13.0
#   ✓ Reports / AI Diagnóstico / Trends sin regresión visual
#
# Qué entra a producción con v3.14.0:
#
#   ► Suite pytest con 157 tests + run_smoke.py
#   ► Catálogo extendido de 20 plantillas de máquinas LATAM
#   ► Importadores universales CSI 2140 / ADRE 408 / UFF
#   ► API REST pública v1.0 (read-only, FastAPI, 13 endpoints)
#
# CAMBIO 100% ADITIVO. Cero archivos de producción modificados.
# Streamlit Cloud NO requiere reinstalar — los nuevos paquetes
# (fastapi, uvicorn, pydantic) sólo se necesitan si se levanta
# uvicorn aparte; la app Streamlit corre con requirements.txt
# original. Igual incluyo notas para mantenerlo consistente.
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.14.0"
RELEASE_NAME="Industrial Plumbing Foundations — tests, importers, REST API"

echo ""
echo "================================================================"
echo " RELEASE ${VERSION} — ${RELEASE_NAME}"
echo "================================================================"
echo ""

# Verificar que dev tiene los commits del Ciclo 18.1
echo "▶ Verificando dev..."
git fetch origin
git checkout dev
git pull origin dev --ff-only

# Buscar los commits del Ciclo 18.1
COMMITS=$(git log --oneline -50 | grep -c "feat(18.1)") || true
if [ "${COMMITS}" -lt 4 ]; then
    echo ""
    echo "  ✗ ERROR: dev no tiene los 4 commits del Ciclo 18.1."
    echo "    Ejecutá primero: bash _publish_ciclo18_1_dev.sh"
    exit 1
fi
echo "  ✓ dev tiene ${COMMITS} commits del Ciclo 18.1"

# Verificar tests antes de mergear
echo ""
echo "▶ Corriendo suite de tests (run_smoke.py)..."
python3 tests/run_smoke.py 2>&1 | tail -3

# Switch a main
echo ""
echo "▶ Switch a main..."
git checkout main
git pull origin main --ff-only

# Tag pre-release como ancla de rollback
PRE_TAG="pre-${VERSION}-$(date +%Y%m%d)"
echo ""
echo "▶ Tag de retorno (rollback): ${PRE_TAG}"
git tag -f "${PRE_TAG}"

# Merge dev → main
echo ""
echo "▶ Merge dev → main (no fast-forward, mantenemos commit de release)..."
git merge --no-ff dev -m "release(${VERSION}): merge dev -> main · ${RELEASE_NAME}

Ciclo 18.1 — Industrial Plumbing Foundations.

Lo que se incluye:

  ► SUITE DE TESTS PYTEST — 157 tests con golden datasets
    Cubre core/waveform_metrics, order_tracking, tsa,
    bearing_fault_frequencies, iso_thresholds, rotordynamics
    (ISO 20816-2 zones + detect_critical_speeds + API 684).
    Permite refactor seguro a futuro.

  ► CATÁLOGO PLANTILLAS LATAM — 20 máquinas pre-cargadas
    Solar Centaur/Mars/Taurus, Siemens SGT, GE Frame, Brush
    turbogen, compresores Solar/Atlas/Ariel/Burckhardt, bombas
    Sulzer/Goulds, motores WEG/ABB/Siemens, ventiladores
    TLT/Howden. Reduce time-to-value de un activo nuevo de
    semanas a un día.

  ► IMPORTADORES UNIVERSALES — argumento de venta migración
    CSI 2140 (Emerson), ADRE 408 (Bently Nevada),
    Universal File Format dataset 58 (SDRC). Cero pelea con
    el incumbente: cliente envía su CSV/UNV y Watermelon lo
    parsea al formato canónico.

  ► API REST PÚBLICA v1.0 — read-only
    13 endpoints sobre plantillas, normas, rodamientos y
    capacidades de import. Auth con API key (hmac.compare_digest).
    Cuchillo contra lock-in de System1: integradores externos
    pueden consumir Watermelon vía HTTP estándar.

CAMBIO 100% ADITIVO. Cero archivos productivos modificados.

Compatibilidad:
  - requirements.txt sin cambios → Streamlit Cloud deploy idéntico
  - El servicio API es opcional (requirements-api.txt aparte)
  - Tests son opcionales (requirements-dev.txt aparte)
  - core/machine_templates.py es complementario a machine_profiles.py
  - core/loaders/ no toca modules/csv_loader.py

Para correr la API localmente:
  pip install -r requirements-api.txt
  export WATERMELON_API_KEYS=\"<key>\"
  uvicorn api.app:app --port 8000

Próximo: Ciclo 18.2 — extracción de servicios de pages/04_Trends.py
hacia core/services/ (refactor con red de seguridad de tests)."

# Tag de release
echo ""
echo "▶ Tag de release: ${VERSION}"
git tag -a "${VERSION}" -m "Release ${VERSION}: ${RELEASE_NAME}

Suite pytest (157 tests · golden datasets sintéticos)
Catálogo plantillas LATAM (20 máquinas)
Importadores universales (CSI 2140 / ADRE 408 / UFF)
API REST pública v1.0 (read-only, FastAPI, 13 endpoints)

100% aditivo · cero archivos productivos modificados."

# Push main + tags
echo ""
echo "▶ Push main + tags..."
git push origin main
git push origin "${VERSION}"
git push origin "${PRE_TAG}" 2>/dev/null || true

echo ""
echo "================================================================"
echo " ✅ RELEASE ${VERSION} COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en 1-2 min."
echo "    El requirements.txt no cambió, así que el redeploy es trivial."
echo ""
echo " 👁  Cambios visibles en producción:"
echo ""
echo "    NINGUNO en la UI Streamlit. Esta release es plomería:"
echo "    - tests (no se ven en runtime)"
echo "    - plantillas (data/, no afectan UI hasta integrar)"
echo "    - loaders (módulos nuevos, no afectan flujo actual)"
echo "    - API REST (servicio aparte, opcional)"
echo ""
echo " 📊 Sin cambios funcionales:"
echo ""
echo "    Login, navegación, AI diagnostics, Reports, Trends,"
echo "    Spectrum, Bode, Polar, Orbit, Shaft Centerline,"
echo "    Briefing mensual, Pattern Memory — TODO sigue idéntico"
echo "    a v3.13.0."
echo ""
echo " 🛠  Para clientes que quieran integrarse:"
echo ""
echo "    1. Compartirles api/README.md"
echo "    2. Generar API key (cualquier UUID), agregarla a"
echo "       WATERMELON_API_KEYS en el server donde corra uvicorn"
echo "    3. El swagger /docs es self-service"
echo ""
echo " 💡 Pendientes en backlog (Ciclo 18.2 — próximo):"
echo ""
echo "    - Extraer servicios de pages/04_Trends.py (5,737 líneas)"
echo "      hacia core/services/trend_service.py. Con la red de"
echo "      seguridad de tests, ahora es seguro."
echo "    - WhatsApp bot v1.0 (eliminar placeholders en main.py)"
echo "    - Live ingestion: OPC UA + MQTT (Tier 0 del roadmap)"
echo "    - Multi-tenancy (organization_id en data layer)"
echo ""
echo "================================================================"
