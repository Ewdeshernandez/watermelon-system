#!/bin/bash
# =============================================================
# Watermelon — Release v3.15.0: Importers/Templates UI Hub
# =============================================================
# Pre-requisitos:
#   ✓ _publish_ciclo18_2_dev.sh ya ejecutado
#   ✓ pages/17_Importers.py corre OK en streamlit run app.py
#   ✓ Reports / AI Diagnóstico / Trends sin regresión
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.15.0"
RELEASE_NAME="Importers & Templates UI Hub"

echo ""
echo "================================================================"
echo " RELEASE ${VERSION} — ${RELEASE_NAME}"
echo "================================================================"

git fetch origin
git checkout dev
git pull origin dev --ff-only

COMMITS=$(git log --oneline -50 | grep -c "feat(18.2)") || true
if [ "${COMMITS}" -lt 1 ]; then
    echo "  ✗ ERROR: dev no tiene commit del Ciclo 18.2."
    echo "    Ejecutá primero: bash _publish_ciclo18_2_dev.sh"
    exit 1
fi
echo "  ✓ dev tiene ${COMMITS} commit(s) del Ciclo 18.2"

echo "▶ Tests..."
python3 tests/run_smoke.py 2>&1 | tail -3

echo "▶ Switch a main..."
git checkout main
git pull origin main --ff-only

PRE_TAG="pre-${VERSION}-$(date +%Y%m%d)"
echo "▶ Tag de retorno: ${PRE_TAG}"
git tag -f "${PRE_TAG}"

echo "▶ Merge dev → main..."
git merge --no-ff dev -m "release(${VERSION}): merge dev -> main · ${RELEASE_NAME}

Ciclo 18.2 — Importers & Templates UI Hub.

Página nueva pages/17_Importers.py con dos tabs:

  ► IMPORTAR CSV (CSI 2140 / ADRE 408 / UFF)
    File uploader + selector de vendor + preview gráfico.
    Botón 'Cargar como Signal Watermelon' inyecta el archivo
    parseado al session_state['signals'] para que Spectrum,
    Trends, Orbit y demás lo vean igual que un load nativo.
    Argumento de venta directo: cliente que viene de System1
    o AMS Suite no tiene que tocar su data.

  ► PLANTILLAS LATAM (20 máquinas pre-cargadas)
    Solar Centaur/Mars/Taurus, Siemens SGT, GE Frame, Brush,
    compresores Solar/Atlas/Ariel/Burckhardt, bombas Sulzer/
    Goulds, motores WEG/ABB/Siemens, ventiladores TLT/Howden.
    Detalle completo: RPM, rodamientos, normas ISO/API,
    sensores recomendados, notas técnicas.

CAMBIO ADITIVO. Modificaciones a páginas productivas: NINGUNA.

Si esta página causa cualquier issue, se borra el archivo
pages/17_Importers.py y la app vuelve a v3.14.0 sin secuelas."

echo "▶ Tag ${VERSION}..."
git tag -a "${VERSION}" -m "Release ${VERSION}: ${RELEASE_NAME}"

echo "▶ Push main + tags..."
git push origin main
git push origin "${VERSION}"

echo ""
echo "================================================================"
echo " ✅ RELEASE ${VERSION} COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud redeploya wm-home-final-2026 en 1-2 min."
echo " 👁  Cambio visible: aparece '17 Importers' en el sidebar."
echo " 📊 Reports / AI / Trends sin cambios."
echo ""
echo "================================================================"
