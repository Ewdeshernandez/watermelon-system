#!/bin/bash
# Release v3.17.0: Crear activo desde plantilla (auto-relleno editable).
set -e
cd "$(dirname "$0")"

VERSION="v3.17.0"
RELEASE_NAME="Create asset from LATAM template (1-click + editable)"

echo "================================================================"
echo " RELEASE ${VERSION}"
echo "================================================================"

git fetch origin
git checkout dev
git pull origin dev --ff-only

echo "▶ Tests..."
python3 tests/run_smoke.py 2>&1 | tail -3

git checkout main
git pull origin main --ff-only

PRE_TAG="pre-${VERSION}-$(date +%Y%m%d)"
git tag -f "${PRE_TAG}"

# Bump VERSION file para que el footer muestre v3.17.0
echo "v3.17.0" > VERSION
git add VERSION
git commit -m "chore: bump VERSION to v3.17.0" --allow-empty

git merge --no-ff dev -m "release(${VERSION}): merge dev -> main · ${RELEASE_NAME}

Ciclo 18.3 — crear activo desde plantilla LATAM con auto-relleno
editable.

  ► core/machine_templates.py:
    suggest_profile_key_for_template() — mapeo categoria + RPM →
    profile_key del catálogo legacy MACHINE_PROFILES.

  ► pages/17_Importers.py (Tab Plantillas):
    Expander 'Crear activo desde esta plantilla' con form
    pre-llenado (profile sugerido, notas con metadata de la
    plantilla). Todo editable. Llama core.instance_state.create_instance,
    misma función que Machinery Library.

  ► 8 tests nuevos.

Archivos modificados:
  - core/machine_templates.py  (suma 1 función + 1 export)
  - pages/17_Importers.py       (suma 1 expander al final del Tab 2)
  - VERSION                     (v3.17.0)
Archivos nuevos:
  - tests/test_template_profile_suggestion.py

Reports / AI / Trends / Bode / Polar / Briefing: SIN CAMBIOS."

git tag -a "${VERSION}" -m "Release ${VERSION}: ${RELEASE_NAME}"
git push origin main
git push origin "${VERSION}"

echo ""
echo "================================================================"
echo " ✅ RELEASE ${VERSION} COMPLETADO"
echo "================================================================"
echo " ⏱  Streamlit Cloud redeploya en 1-2 min."
echo " 👁  En wm-home-final-2026:"
echo "    Footer → v3.17.0"
echo "    Importers & Plantillas → Tab Plantillas → Crear activo"
echo "================================================================"
