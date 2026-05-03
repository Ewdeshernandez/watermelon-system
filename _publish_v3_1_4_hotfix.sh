#!/bin/bash
# =============================================================
# Watermelon — v3.1.4 hotfix: HD export REESCRITO MINIMAL
# =============================================================
# Tras 3 intentos (v3.1.1, v3.1.2, v3.1.3) que NO resolvieron
# completamente el bug del PNG HD vacío, voy con approach
# radicalmente simple:
#
# BORRO toda la complejidad de _build_export_safe_figure +
# _scale_export_figure (que hacían múltiples recreate's de
# figura plana perdiendo subplots structure) y reescribo
# build_export_png_bytes desde cero con apenas 3 pasos:
#
#   1. fig.to_dict() — clone preservando subplots+secondary_y
#   2. scattergl→scatter en cada trace (kaleido no soporta WebGL)
#   3. update_layout con width/height/font scaling para HD
#
# Sin scaling per-trace, sin recreate de Figure, sin
# manipulaciones de domain/overlay del axis. La figura sale
# casi como se ve en pantalla, solo más grande.
#
# Las funciones _build_export_safe_figure y _scale_export_figure
# QUEDAN definidas pero ya no se usan — las dejo por
# compatibilidad si algún otro código las llama.
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.1.4"
RELEASE_TITLE="Hotfix: HD export reescrito MINIMAL — solo clone+scattergl+bump fonts"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

echo "▶ Stageando..."
git add pages/04_Trends.py
git add _publish_v3_1_4_hotfix.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commit..."
    git commit -m "fix(trend): HD export reescrito MINIMAL — borradas helpers complejas (17.8.4)

Tras 3 intentos (v3.1.1/2/3) que no resolvieron el bug del PNG
HD sin curvas, voy radicalmente simple:

build_export_png_bytes ahora hace SOLO 3 cosas:
  1. fig.to_dict() — clone preservando subplots+secondary_y
  2. scattergl->scatter en traces (kaleido no soporta WebGL)
  3. update_layout con width/height/font para HD legible

Sin scaling per-trace, sin recreate de Figure, sin
manipulaciones de domain/overlay. La figura sale casi como se
ve en pantalla, solo más grande.

Las helpers _build_export_safe_figure y _scale_export_figure
quedan definidas pero ya no se usan." || echo "  (sin cambios)"
else
    echo "  (no hay cambios)"
fi

echo "▶ Reconciliando..."
git fetch origin dev
git pull --rebase origin dev || { echo "✗ Rebase falló."; exit 1; }

echo "▶ Push dev..."
git push origin dev

echo "▶ Switch a main..."
git checkout main
git pull origin main

echo "▶ Merge dev → main..."
git merge dev --no-ff -m "hotfix(${VERSION}): ${RELEASE_TITLE}

Sobre v3.1.3 (que tampoco resolvio el HD vacio):

Reescribi build_export_png_bytes con approach radicalmente
simple: solo to_dict clone + scattergl-to-scatter + bump
de fonts/dimensions. Sin las helpers complejas que recreaban
figuras planas y perdian subplots structure."

echo "▶ Tag ${VERSION}..."
git tag -a "${VERSION}" -m "${RELEASE_TITLE}

Hotfix sobre v3.1.3 (que no funciono): HD export del Trend
ahora usa pipeline minimal de 3 pasos. Si esto no funciona
es bug de kaleido/plotly, no de la transformacion."

echo "▶ Push main + tag..."
git push origin main
git push origin "${VERSION}"

echo "▶ Vuelta a dev..."
git checkout dev

echo ""
echo "================================================================"
echo " ✓ ${VERSION} publicado a main"
echo "================================================================"
echo ""
echo " Refrescá Cmd+Shift+R y probá Prepare PNG HD → Download."
echo ""
echo " Si TODAVÍA sale vacío después de esto, es bug de"
echo " kaleido o de la versión de plotly. Sería el momento de"
echo " probar to_image con engine='kaleido' explícito o downgradear"
echo " plotly. Avisame y lo investigamos."
echo "================================================================"
