#!/bin/bash
# =============================================================
# Watermelon — Ciclo 18.2 → DEV: Importers/Templates UI Hub
# =============================================================
# Enchufa a la UI las capacidades del Ciclo 18.1.
#
# Lo nuevo:
#   ► pages/17_Importers.py — página NUEVA (no modifica ninguna existente)
#     Tab 1: subir CSV de CSI 2140 / ADRE 408 / UFF, parsear, ver
#            preview, e inyectar al session_state["signals"] para que
#            Spectrum/Trends/etc. lo vean igual que un load nativo.
#     Tab 2: navegador de las 20 plantillas LATAM con metadata
#            completa (RPM, rodamientos, normas, sensores).
#
# Cambios técnicos:
#   (NUEVO) pages/17_Importers.py
#   Modificados productivos: NINGUNO.
#
# Si algo se rompe, basta con: rm pages/17_Importers.py
# =============================================================

set -e
cd "$(dirname "$0")"

PRE_TAG="pre-ciclo18-2-$(date +%Y%m%d)"
echo "▶ Tag de retorno: ${PRE_TAG}"
git tag -f "${PRE_TAG}"

echo "▶ Switch a dev..."
if git show-ref --verify --quiet refs/heads/dev; then
    git checkout dev
else
    git checkout -b dev origin/dev
fi
git pull origin dev --ff-only || echo "  (sin cambios remotos)"

echo "▶ Branch feat..."
git checkout -B feat/ciclo18-2-importers-ui

echo "▶ Commit..."
git add pages/17_Importers.py
git commit -m "feat(18.2): página Importadores & Plantillas LATAM en UI

Nueva página pages/17_Importers.py con dos tabs:

Tab 1 — Importar CSV (CSI 2140 / ADRE 408 / UFF):
  - File uploader + selector de vendor
  - Parser via core.loaders.* (Ciclo 18.1)
  - Preview gráfico (time o spectrum) con Plotly
  - Metric cards: muestras, dominio, fs, rpm
  - Metadata cruda en expander
  - Botón 'Cargar como Signal Watermelon' → inyecta a
    st.session_state['signals'] para que Spectrum/Trends/etc.
    lo vean igual que un load nativo.

Tab 2 — Plantillas LATAM:
  - Filtro por categoría
  - Selector de plantilla (20 disponibles)
  - Detalle: fabricante, RPM nominal, rango, potencia,
    rodamientos típicos, normas ISO/API, esquema de sensores,
    notas técnicas.

Auth: require_login + require_role(admin, specialist).
Page slot: 17 (entre 16_Reports y 18+).

ZERO modificaciones a páginas existentes. Si esta página causa
algún issue, se borra el archivo y la app vuelve a v3.14.0." || echo "  (sin cambios)"

echo "▶ Push feat..."
git push -u origin feat/ciclo18-2-importers-ui

echo "▶ Merge → dev..."
git checkout dev
git merge --no-ff feat/ciclo18-2-importers-ui -m "Merge feat/ciclo18-2-importers-ui into dev

Ciclo 18.2 — wiring de importadores universales y plantillas LATAM
a la UI Streamlit (página nueva pages/17_Importers.py).
Sin modificaciones a páginas productivas existentes."

echo "▶ Push dev..."
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 18.2 en DEV"
echo "================================================================"
echo ""
echo " Verificación:"
echo "   1. streamlit run app.py"
echo "   2. Login → debe aparecer '17 Importers' en el sidebar"
echo "   3. Click → 2 tabs (Importar / Plantillas)"
echo "   4. Resto de páginas IGUAL que v3.14.0"
echo ""
echo " Si OK → bash _publish_v3_15_0_to_main.sh"
echo " Si rompe → rm pages/17_Importers.py && git checkout pages/"
echo "================================================================"
