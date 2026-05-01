#!/bin/bash
# =============================================================
# Watermelon — HOTFIX Ciclo 15.1: render_user_menu duplicado (dev)
# =============================================================
# Bug: pages/01b_Machine_Map.py llamaba render_user_menu() DOS veces:
# una al inicio del archivo (siguiendo patron viejo) y otra antes de
# page_header(). Streamlit detecta los botones duplicados y lanza:
#
#   StreamlitDuplicateElementKey: There are multiple elements with the
#   same key='nav_pages/_landing.py'.
#
# Fix: dejar UNA sola llamada a render_user_menu() despues de
# st.set_page_config + require_login (mismo patrón que 00_Machinery_Library
# y otras páginas).
#
# Tambien removida la llamada redundante a require_login() al inicio
# del archivo.
#
# Ejecutar:
#   bash _hotfix_ciclo15_1_dup_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add pages/01b_Machine_Map.py
git status --short | head

git commit -m "fix(machine-map): hotfix Ciclo 15.1 — render_user_menu duplicado

pages/01b_Machine_Map.py llamaba render_user_menu() dos veces, lo que
causaba StreamlitDuplicateElementKey en los botones del menu lateral.

Fix: una sola llamada despues de set_page_config + require_login,
mismo patron que las demas paginas. Removida tambien la llamada
redundante a require_login() al inicio del archivo."

git push origin dev

echo ""
echo "Refrescar app — la pagina Machine Map ya carga sin error."
