#!/bin/bash
# =============================================================
# Watermelon — Release v3.16.0: Importers UI + Version Display Fix
# =============================================================
# Pre-requisitos:
#   ✓ Todos los hotfixes Ciclo 18.2 ya en dev:
#     - feat(18.2): página pages/17_Importers.py
#     - fix(18.2): wiring NAV (Importers en sidebar)
#     - fix: archivo VERSION + bump _FALLBACK_VERSION
#     - fix: prioridad VERSION > git_latest_tag (v3.16.0 visible)
#   ✓ wm-test.streamlit.app muestra v3.16.0 sin errores
#
# Lo que entra a producción con v3.16.0:
#
#   ► PÁGINA NUEVA pages/17_Importers.py
#     Tab 1: importar CSV de CSI 2140 / ADRE 408 / UFF y convertir a
#            formato Watermelon. Botón inyecta a session_state.
#     Tab 2: catálogo de las 20 plantillas LATAM con metadata
#            completa (RPM, rodamientos, normas, sensores).
#
#   ► WIRING NAV (core/auth.py)
#     'Importers & Plantillas' aparece en el sidebar entre Reports
#     y AI Assistant. CLIENT_BLOCKED_PAGES incluye 17_Importers.
#
#   ► FIX VERSION DISPLAY
#     Streamlit Cloud hace shallow clone que pierde tags v3.x
#     recientes. Se invierte prioridad: VERSION file > git_latest_tag.
#     Archivo VERSION = source of truth declarativa (v3.16.0).
#     _FALLBACK_VERSION subido a v3.15.0 (era v3.0.8 obsoleto).
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.16.0"
RELEASE_NAME="Importers UI + Version Display Fix"

echo "================================================================"
echo " RELEASE ${VERSION} — ${RELEASE_NAME}"
echo "================================================================"

git fetch origin
git checkout dev
git pull origin dev --ff-only

echo "▶ Tests..."
python3 tests/run_smoke.py 2>&1 | tail -3

git checkout main
git pull origin main --ff-only

PRE_TAG="pre-${VERSION}-$(date +%Y%m%d)"
echo "▶ Tag de retorno: ${PRE_TAG}"
git tag -f "${PRE_TAG}"

echo "▶ Merge dev → main..."
git merge --no-ff dev -m "release(${VERSION}): merge dev -> main · ${RELEASE_NAME}

Ciclo 18.2 + hotfixes — Importers UI Hub + version display fix.

Lo que se incluye:

  ► PÁGINA NUEVA: pages/17_Importers.py
    Tab 1 — Importar CSV (CSI 2140 / ADRE 408 / UFF):
      File uploader + selector de vendor + preview gráfico.
      Botón 'Cargar como Signal Watermelon' inyecta a
      st.session_state['signals'] para que Spectrum / Trends /
      Orbit lo vean igual que un load nativo.
    Tab 2 — Plantillas LATAM (20 máquinas pre-cargadas):
      Solar / Siemens / GE / Brush / Atlas Copco / Ariel /
      Burckhardt / Sulzer / Goulds / WEG / ABB / SIMOTICS /
      TLT / Howden con RPM, rodamientos, normas ISO/API,
      sensores recomendados, notas técnicas.

  ► WIRING NAV (core/auth.py):
    'Importers & Plantillas' visible en sidebar.
    Bloqueada para role=client.

  ► FIX VERSION DISPLAY:
    Streamlit Cloud hace shallow clone que NO incluye tags v3.x
    del repo. Esto provocaba que el footer mostrara v2.1
    (tag más viejo que sí existía en el clone). Solución:
      - Archivo VERSION agregado al repo (source of truth declarativa).
      - core/version.py: prioridad invertida (VERSION > git_latest_tag).
      - _FALLBACK_VERSION bumpeado de v3.0.8 a v3.15.0.

Archivos productivos modificados en este release:
  - core/auth.py      (1 entrada NAV + 1 entrada CLIENT_BLOCKED_PAGES)
  - core/version.py   (lógica de prioridad)
Archivos nuevos:
  - VERSION
  - pages/17_Importers.py

Reports / AI Diagnostics / Trends / Spectrum / Bode / Polar /
Orbit / Briefing / Pattern Memory: SIN CAMBIOS."

echo "▶ Tag ${VERSION}..."
git tag -a "${VERSION}" -m "Release ${VERSION}: ${RELEASE_NAME}"

echo "▶ Push main + tags..."
git push origin main
git push origin "${VERSION}"

echo ""
echo "================================================================"
echo " ✅ RELEASE ${VERSION} COMPLETADO"
echo "================================================================"
echo " ⏱  Streamlit Cloud redeploya wm-home-final-2026 en 1-2 min."
echo " 👁  Cambios visibles:"
echo "    - Footer login dice v3.16.0 (antes v2.1)"
echo "    - Sidebar tiene 'Importers & Plantillas'"
echo " 📊 Resto de páginas: idénticas a v3.14.0."
echo "================================================================"
