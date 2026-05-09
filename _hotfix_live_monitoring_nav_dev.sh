#!/bin/bash
# Hotfix → DEV: Registrar Live Monitoring en NAV_ITEMS.
# La página existía pero no aparecía en el sidebar porque el menú custom de
# core/auth.py lista las páginas explícitamente (no usa el sidebar default
# de Streamlit).
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B hotfix/live-monitoring-nav

git add core/auth.py VERSION
git commit -m "fix(23.1.1): registrar Live Monitoring en NAV_ITEMS

  Síntoma: después del deploy de Tier 0 A, la página
  pages/02_Live_Monitoring.py no aparecía en el sidebar a pesar de existir
  en disco.

  Causa: core/auth.py.NAV_ITEMS define el menú custom explícito; cualquier
  página nueva debe agregarse acá manualmente.

  Fix: nueva entrada '🔴 Live Monitoring' después de Tabular List, en la
  posición lógica del flujo (valores en vivo → tabular → mapa de máquina).

VERSION → v3.31.1-dev"

git push -u origin hotfix/live-monitoring-nav
git checkout dev
git merge --no-ff hotfix/live-monitoring-nav -m "Merge hotfix/live-monitoring-nav into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Hotfix en DEV — v3.31.1-dev"
echo " Refrescá wm-test.streamlit.app — debería aparecer"
echo "  '🔴 Live Monitoring' debajo de Tabular List."
echo "================================================================"
