#!/bin/bash
# =============================================================
# Watermelon — v2.4 → MAIN: cleanup expander de debug
# =============================================================
# Pequeño bump sobre v2.3. Trae a main el único hotfix pendiente
# en dev:
#
#   - pages/01__Tabular_List.py: removido el expander
#     "🔍 Debug: matching de sensores" que ya cumplió su función
#     (encontramos el bug del falso match cross-tipo con '*4*x*'
#     vs '64x' del oversampling Bently). La página de Tabular
#     List queda 100% limpia para producción.
#
# Si en algún caso futuro el matcher cae mal y hace falta volver
# a inspeccionar, el expander se restaura desde el historial git
# (commit con BUILD 14c.3-debug-v2).
#
# El banner amarillo "Override criterio activo" NO es debug —
# ese es legítimo cuando el usuario tiene valores manuales en
# el sidebar avanzado y se queda como está.
#
# Ejecutar:
#   bash _publish_v2_4_to_main.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " STEP 0: Verificar branch y estado"
echo "================================================================"
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "dev" ]; then
  git checkout dev
fi
git pull origin dev || true
git status --short | grep -vE '^\?\?' | head

echo ""
echo "================================================================"
echo " STEP 1: Tag pre-merge en main para rollback"
echo "================================================================"
git fetch origin
PRE_MERGE_TAG="v2.4-pre-main-$(date +%Y%m%d-%H%M%S)"
git tag -a "$PRE_MERGE_TAG" origin/main -m "Snapshot de main antes del merge v2.4"
git push origin "$PRE_MERGE_TAG"
echo "  Tag de rollback creado: $PRE_MERGE_TAG"

echo ""
echo "================================================================"
echo " STEP 2: Merge dev → main"
echo "================================================================"
git checkout main
git pull origin main
git merge --no-ff dev -m "merge: dev → main — Watermelon v2.4

Cleanup sobre v2.3:

* pages/01__Tabular_List.py: removido el expander
  '🔍 Debug: matching de sensores'. Ya cumplio su funcion
  (encontramos el bug del falso match cross-tipo con
  '*4*x*' vs '64x' del oversampling Bently en v2.3).
  La pagina de Tabular List queda limpia para produccion.

Si en algun caso futuro el matcher cae mal, restaurar el
expander desde el historial git (commit con
BUILD 14c.3-debug-v2)."

echo ""
echo "================================================================"
echo " STEP 3: Tag v2.4 y push"
echo "================================================================"
git tag -a "v2.4" -m "Watermelon v2.4 — cleanup expander de debug en Tabular List"
git push origin main
git push origin v2.4

echo ""
echo "================================================================"
echo " STEP 4: Volver a dev"
echo "================================================================"
git checkout dev

echo ""
echo "================================================================"
echo " ✓ MERGE A MAIN COMPLETADO — v2.4 LIVE"
echo "================================================================"
echo ""
echo "Tags creados:"
echo "  - $PRE_MERGE_TAG (rollback)"
echo "  - v2.4 (release)"
echo ""
echo "ROLLBACK:"
echo "  git checkout main && git reset --hard $PRE_MERGE_TAG && git push --force-with-lease origin main"
echo ""
echo "Streamlit Cloud va a redeployar en 1-2 min desde main."
echo "Verificar en producción: la página Tabular List ya no debe"
echo "mostrar el expander '🔍 Debug: matching de sensores'."
echo "================================================================"
