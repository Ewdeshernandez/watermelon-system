#!/bin/bash
# =============================================================
# Watermelon — HOTFIX Ciclo 14a: navegación rota tras renombrado (dev)
# =============================================================
# El Ciclo 14a renombró pages/17_Asset_Documents.py → 00_Machinery_Library.py
# pero el menú de navegación en core/auth.py NAV_ITEMS seguía apuntando
# al archivo viejo. Cualquier click en otra página después del rename
# disparaba:
#
#   StreamlitAPIException: Could not find page:
#   pages/17_Asset_Documents.py
#
# Fix:
# - core/auth.py NAV_ITEMS: "Asset Documents" → "Machinery Library"
#   con path actualizado a pages/00_Machinery_Library.py.
# - Reordenado: Machinery Library queda como segundo del menú,
#   justo después de Home (flujo correcto: primero seleccionar
#   máquina activa, después cargar CSVs en Load Data).
# - Docstring del propio archivo actualizado (cosmético).
#
# Ejecutar:
#   bash _hotfix_ciclo14a_nav_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock

CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "dev" ]; then
  echo "Cambiando a dev..."
  git checkout dev
fi
git pull origin dev

echo "[1] Adoptando hotfix..."
git add core/auth.py pages/00_Machinery_Library.py
git status --short | head
echo ""

echo "[2] Commit..."
git commit -m "fix(nav): hotfix Ciclo 14a — actualizar NAV_ITEMS al nuevo path

El renombrado pages/17_Asset_Documents.py → 00_Machinery_Library.py
dejó core/auth.py NAV_ITEMS apuntando al path viejo, lo que rompía
la navegación con StreamlitAPIException 'Could not find page'.

Fix:
* NAV_ITEMS: 'Asset Documents' → 'Machinery Library' con path
  actualizado a pages/00_Machinery_Library.py.
* Reordenado: Machinery Library queda como 2do del menu, justo
  despues de Home. Flujo natural: primero elegir maquina activa,
  despues cargar CSVs en Load Data.
* Docstring del archivo actualizado (cosmético)."
echo "    OK"

echo "[3] Push..."
git push origin dev
echo "    OK"

echo ""
echo "================================================================"
echo " HOTFIX aplicado"
echo "================================================================"
echo ""
echo "Refrescar la app (Ctrl+R o re-deploy en Streamlit Cloud) y"
echo "volver a probar el flujo del Ciclo 14a desde Sección A."
