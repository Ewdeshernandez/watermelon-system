#!/bin/bash
# =============================================================
# Watermelon — v3.1.1 hotfix URGENTE: Trend x-axis muestra ns
# =============================================================
# BUG CRÍTICO REPORTADO EN MAIN/PRODUCCIÓN:
# El eje X del Trend chart muestra timestamps en NANOSEGUNDOS
# en notación científica (1.7765×10^18, 1.7766×10^18) en lugar
# de fechas legibles (2026-04-18, 2026-04-19...). Reporte
# inutilizable para el cliente.
#
# CAUSA:
# En mixed mode (vibración + operacional con secondary_y),
# Plotly a veces NO infiere el x-axis como 'date' aunque los
# datos sean datetime64[ns]. Cae a 'linear' y los renderea como
# números crudos en nanosegundos. Triggered probablemente por
# alguna combinación de make_subplots + secondary_y + traces
# con dtypes mezclados.
#
# FIX:
# Forzar EXPLÍCITAMENTE type="date" en TODOS los xaxis configs
# del trend figure:
#   - update_xaxes() del mixed mode (secondary_y branch)
#   - xaxis=dict(...) del single-axis layout
#   - update_xaxes() del HD export figure
# Plus tickformat="%Y-%m-%d %H:%M" y hoverformat detallado para
# que las labels y los tooltips siempre se vean bien.
#
# Va DIRECTO a main porque es bug que rompe la generación de
# reportes profesionales para el cliente.
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.1.1"
RELEASE_TITLE="Hotfix urgente: Trend x-axis date type forzado (no más nanosegundos)"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

echo ""
echo "▶ Stageando hotfix..."
git add pages/04_Trends.py
git add _publish_v3_1_1_hotfix.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commiteando..."
    git commit -m "fix(trend): x-axis muestra timestamps en ns — forzar type=date (17.8.1)

Bug critico en main: el eje X del Trend chart muestra
timestamps en nanosegundos (1.7765×10^18) en notacion
cientifica en lugar de fechas. Reporte inutilizable.

Causa: en mixed mode con secondary_y, Plotly a veces no
infiere el x-axis como 'date' aunque los datos sean
datetime64[ns]. Cae a 'linear' y renderea numeros crudos.

Fix: forzar explicitamente type='date' en TODOS los xaxis
configs del trend figure (mixed mode update_xaxes, single-
axis layout xaxis=dict, HD export update_xaxes). Plus
tickformat='%Y-%m-%d %H:%M' y hoverformat detallado." || echo "  (sin cambios)"
else
    echo "  (no hay cambios)"
fi

echo ""
echo "▶ Reconciliando..."
git fetch origin dev
git pull --rebase origin dev || { echo "✗ Rebase falló."; exit 1; }

echo "▶ Push dev..."
git push origin dev

echo ""
echo "▶ Switch a main..."
git checkout main
git pull origin main

echo "▶ Merge dev → main..."
git merge dev --no-ff -m "hotfix(${VERSION}): ${RELEASE_TITLE}

Sobre v3.1.0:

Bug critico: Trend chart mostraba timestamps en nanosegundos
(1.7765×10^18) en lugar de fechas legibles. Plotly caia a
linear axis en mixed mode con secondary_y. Fix: type='date'
forzado en todos los xaxis configs + tickformat YYYY-MM-DD HH:MM."

echo "▶ Tag ${VERSION}..."
git tag -a "${VERSION}" -m "${RELEASE_TITLE}

Hotfix sobre v3.1.0: x-axis del Trend ahora siempre
formatea como fecha (YYYY-MM-DD HH:MM), nunca como
timestamp numerico en nanosegundos."

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
echo " Streamlit Cloud auto-redeploya en ~2 min. Refresca Cmd+Shift+R."
echo " El eje X del Trend ahora va a mostrar fechas como"
echo " '2026-04-18 00:00' en lugar de '1.7765×10^18'."
echo "================================================================"
