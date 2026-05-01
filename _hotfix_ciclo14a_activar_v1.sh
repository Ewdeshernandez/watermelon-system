#!/bin/bash
# =============================================================
# Watermelon — HOTFIX 3 Ciclo 14a: boton Activar no responde (dev)
# =============================================================
# Boton "Activar" en la card del grid no respondia: se cliqueaba,
# parecia que pasaba algo, pero la card seguia mostrando "Activar"
# en vez de "✓ activa".
#
# Causa: el sidebar selectbox (render_instance_selector) tiene su
# propia state key 'wm_instance_select_documents' y se renderiza
# ANTES que el grid en el orden de ejecucion. En cada rerun:
#   1. Selectbox lee 'wm_instance_select_documents' (valor viejo)
#      y SOBRE-ESCRIBE 'wm_active_instance_id' con ese valor viejo.
#   2. El grid hace click "Activar" → setea 'wm_active_instance_id'
#      → st.rerun().
#   3. En el rerun, paso 1 sobreescribe el cambio del paso 2.
#   → bug perpetuo.
#
# Fix: cuando se hace click en "Activar", sincronizar AMBAS keys
# en el mismo callback (la principal + la del selectbox del sidebar)
# antes del st.rerun(). Asi el sidebar al re-renderizarse lee el
# valor nuevo y no lo pisa.
#
# Ejecutar:
#   bash _hotfix_ciclo14a_activar_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add pages/00_Machinery_Library.py
git status --short | head

git commit -m "fix(library): hotfix 3 Ciclo 14a — boton Activar sincroniza state del selectbox

El selectbox del sidebar usa key 'wm_instance_select_documents' separada
de 'wm_active_instance_id', y se renderiza antes del grid. En cada rerun
el selectbox sobreescribia el valor activado por el boton del grid.

Fix: al click en Activar, setear AMBAS keys antes del rerun. El sidebar
al re-renderizarse lee el valor nuevo y respeta la activacion."

git push origin dev

echo ""
echo "Refrescar app y ahora 'Activar' debe funcionar end-to-end."
