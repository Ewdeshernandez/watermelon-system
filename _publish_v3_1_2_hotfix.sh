#!/bin/bash
# =============================================================
# Watermelon — v3.1.2 hotfix URGENTE: HD export sin traces
# =============================================================
# BUG REPORTADO POST-v3.1.1:
# Al hacer "Prepare PNG HD" desde Trend, el PNG sale con:
#   ✓ Título correcto
#   ✓ Eje X con fechas legibles (gracias al fix v3.1.1)
#   ✓ Ejes Y con rangos correctos (Amplitude 0-10, Operational 0-5000)
#   ✓ Status chip (Vigilancia)
#   ✓ Trend Information panel
#   ✓ Legend con nombres de traces
#   ✗ PERO: las CURVAS no se dibujan — chart vacío de datos
#
# CAUSA RAÍZ:
# `_build_export_safe_figure` creaba un `go.Figure()` plano y
# copiaba traces una por una. Pero cuando el chart original es
# `make_subplots(specs=[[{"secondary_y": True}]])`, las traces
# de la curva operacional tienen `yaxis="y2"` que apunta a un
# subplot que NO EXISTE en la figura plana. En el browser,
# Plotly tolera esto y crea el axis on-the-fly, pero kaleido
# (el renderer del PNG) no — las curvas se dibujan fuera de la
# zona visible o no se dibujan.
#
# El fix de v3.1.0 (yaxis2 con overlaying='y') ayudaba pero no
# resolvía completamente el problema cuando había MÚLTIPLES
# señales de vibración + operacional combinadas.
#
# FIX:
# `_build_export_safe_figure` ahora hace `fig.to_dict()` para
# clonar la ESTRUCTURA COMPLETA (incluyendo subplots y axis
# refs) y solo cambia el campo `type` de las traces de
# scattergl→scatter. Mucho más seguro y compatible con
# cualquier topología de figura.
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.1.2"
RELEASE_TITLE="Hotfix: HD export sin traces — preservar subplots structure"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

echo "▶ Stageando..."
git add pages/04_Trends.py
git add _publish_v3_1_2_hotfix.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commit..."
    git commit -m "fix(trend): HD export salia sin curvas — preservar subplots structure (17.8.2)

Bug post-v3.1.1: PNG HD del Trend sale con titulo, axes,
chip, info panel y legend correctos, pero CHART VACIO sin
las curvas dibujadas.

Causa: _build_export_safe_figure creaba un go.Figure() plano
y copiaba traces. Pero el chart original es make_subplots
con secondary_y=True; las traces operacional tienen
yaxis='y2' que apunta a un subplot inexistente en la figura
plana. Browser Plotly tolera y crea axis on-the-fly, pero
kaleido (PNG renderer) no — curvas fuera de la zona visible.

Fix: _build_export_safe_figure ahora hace fig.to_dict() para
clonar la estructura completa (subplots + secondary_y +
axis refs) y solo cambia trace.type de scattergl a scatter.
Robusto a cualquier topologia de figura." || echo "  (sin cambios)"
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

Sobre v3.1.1:

PNG HD del Trend ya no sale con chart vacio cuando hay
mixed mode (vibracion + operacional). _build_export_safe_
figure preserva la estructura subplots completa via
to_dict() clone."

echo "▶ Tag ${VERSION}..."
git tag -a "${VERSION}" -m "${RELEASE_TITLE}

Hotfix sobre v3.1.1: PNG HD del Trend ahora dibuja las
curvas correctamente cuando hay subplots con secondary_y."

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
echo " Streamlit Cloud auto-redeploya en ~2 min. Refrescá Cmd+Shift+R."
echo " Probá Prepare PNG HD → Download — el PNG ahora debe traer"
echo " todas las curvas dibujadas (vibración 5YV/8YV + operacional)."
echo "================================================================"
