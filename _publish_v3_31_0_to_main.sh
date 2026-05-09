#!/bin/bash
# ===========================================================================
# Release v3.31.0 → MAIN (consolidado)
# ---------------------------------------------------------------------------
# Junta los siguientes ciclos acumulados en dev:
#
#   • 22.2c+d  Captured Parameters progreso + Documents Vault grid
#   • 22.1     Tabular List respeta unit_native (bug fix C-200-C)
#   • 22.3     Wizard único camino para crear activos (form legacy eliminado)
#   • 23.1     Tier 0 A — Live Data Ingestion (Modbus → API → Supabase)
#   • 23.1.1   Live Monitoring registrado en NAV_ITEMS
#
# Pre-flight checks:
#   - dev compila (AST parse)
#   - smoke tests pasan
#   - main está al día con remote
# ===========================================================================
set -e
cd "$(dirname "$0")"

VERSION="v3.31.0"

echo "================================================================"
echo " RELEASE ${VERSION} → MAIN"
echo " Consolida 22.2c+d, 22.1, 22.3, 23.1, 23.1.1"
echo "================================================================"
echo ""

# ----- 1. Pre-flight: dev al día y tests OK ---------------------------------
git fetch origin

echo ""
echo "▸ Validando dev..."
git checkout dev
git pull origin dev --ff-only
python3 tests/run_smoke.py 2>&1 | tail -3

echo ""
echo "▸ Validando main..."
git checkout main
git pull origin main --ff-only

# ----- 2. Bump VERSION definitiva (sin -dev) --------------------------------
echo ""
echo "▸ Bumping VERSION → ${VERSION}"
echo "${VERSION}" > VERSION
git add VERSION
git commit -m "chore: bump VERSION to ${VERSION}" --allow-empty

# ----- 3. Merge dev → main -------------------------------------------------
echo ""
echo "▸ Merge dev → main"
git merge --no-ff dev -m "release(${VERSION}): Visual refactor + Tabular fix + Wizard único + Tier 0 A Live

Consolidated cycles:
  • 22.2c — Captured Parameters con barra progreso + chips por categoría
  • 22.2d — Documents Vault grid de cards con filtros e iconos por tipo
  • 22.1  — Tabular List respeta unit_native (fix C-200-C)
  • 22.3  — Wizard único camino para crear activos (form legacy eliminado)
  • 23.1  — Tier 0 A: Live Data Ingestion (Modbus → API → Supabase)
            * core/live_readings.py persistencia
            * api POST /v1/ingest/live
            * collector/ Windows service (NSSM)
            * data/modbus_maps/tes1.json
            * supabase migration live_readings
            * pages/02_Live_Monitoring.py
  • 23.1.1 — Live Monitoring en NAV_ITEMS

Estratégico: vectores 1X/2X (Ampl + Phase) en tiempo real — feature
equivalente cuesta ~80k USD/año en System1/AMS Suite."

# ----- 4. Tag y push -------------------------------------------------------
echo ""
echo "▸ Tag ${VERSION}"
git tag -a "${VERSION}" -m "Release ${VERSION}: Visual refactor + Tier 0 A Live Data Ingestion"

echo ""
echo "▸ Push main + tag"
git push origin main
git push origin "${VERSION}"

# ----- 5. Resumen ----------------------------------------------------------
echo ""
echo "================================================================"
echo " ✅ ${VERSION} EN MAIN"
echo "================================================================"
echo ""
echo " ⚠️  POST-RELEASE — IMPORTANTE:"
echo ""
echo "  1. Cambiá Render watermelon-api branch DE 'dev' A 'main'"
echo "     (Settings → Build → Branch → main → Save)"
echo "     Esto evita que cambios futuros en dev se deployen a la API"
echo "     productiva sin querer."
echo ""
echo "  2. Streamlit Cloud va a redeployar wm-home-final-2026"
echo "     automáticamente (mira la rama main). Esperá ~2 min y refrescá."
echo ""
echo "  3. Verificá en producción:"
echo "     • Footer dice ${VERSION}"
echo "     • Sidebar tiene '🔴 Live Monitoring' y '🧙 Crear activo'"
echo "     • Machinery Library: card 🧙 reemplaza el form legacy"
echo "     • Tabular List con C-200-C: Family coherente con Unit"
echo ""
echo "================================================================"
