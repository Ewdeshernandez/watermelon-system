#!/bin/bash
# =============================================================
# Hotfix Ciclo 18.2 — Wiring NAV: agregar 17_Importers al sidebar
# =============================================================
# El sidebar de Watermelon es manual (showSidebarNavigation=false
# en .streamlit/config.toml). NAV_ITEMS vive en core/auth.py.
# Este hotfix agrega la entrada para que '17 Importers' aparezca
# en el menú lateral como cualquier otra página.
#
# Cambios:
#   - core/auth.py: agrega "Importers & Plantillas" a NAV_ITEMS
#                    (entre Reports y AI Assistant)
#   - core/auth.py: agrega pages/17_Importers.py a CLIENT_BLOCKED_PAGES
#                    (clientes externos no deberían importar data)
#
# Riesgo: BAJO. Cambio declarativo, no toca lógica de auth.
# =============================================================

set -e
cd "$(dirname "$0")"

PRE_TAG="pre-hotfix-18-2-nav-$(date +%Y%m%d)"
git tag -f "${PRE_TAG}"

echo "▶ Switch a dev..."
git checkout dev
git pull origin dev --ff-only

echo "▶ Branch hotfix..."
git checkout -B hotfix/ciclo18-2-nav

git add core/auth.py
git commit -m "fix(18.2): wiring NAV — agregar 17_Importers al sidebar manual

El sidebar de Watermelon usa NAV_ITEMS hardcodeado en core/auth.py
porque .streamlit/config.toml tiene showSidebarNavigation=false.

Esta página nueva (pages/17_Importers.py) no aparecía en el menú
porque NAV_ITEMS no la incluía. Se agrega entre 'Reports' y
'AI Assistant', y se incluye en CLIENT_BLOCKED_PAGES (no debería
ser visible para role=client porque permite subir data y crear
activos).

Cambio declarativo, no toca lógica de auth."

echo "▶ Push hotfix..."
git push -u origin hotfix/ciclo18-2-nav

echo "▶ Merge → dev..."
git checkout dev
git merge --no-ff hotfix/ciclo18-2-nav -m "Merge hotfix/ciclo18-2-nav into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Hotfix NAV en DEV"
echo "================================================================"
echo " Verificar: refresh wm-test.streamlit.app"
echo "  → debe aparecer 'Importers & Plantillas' en el sidebar"
echo " Si OK → bash _publish_v3_15_0_to_main.sh"
echo "================================================================"
