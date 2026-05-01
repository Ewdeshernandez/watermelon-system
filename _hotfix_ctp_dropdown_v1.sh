#!/bin/bash
# =============================================================
# Watermelon — HOTFIX: dropdown click-to-place volvia al keyphasor
# =============================================================
# Dos bugs juntos:
#  (1) El sort de los planos tenia el signo invertido —
#      Keyphasor quedaba primero (indice 0) y rerun siempre
#      caia ahi.
#  (2) Despues de guardar una posicion, st.rerun() reconstruye
#      el selectbox con LABELS DISTINTAS (de "sin posicionar"
#      a "posicionado"). Streamlit busca el valor anterior por
#      label y al no encontrarlo cae al indice 0.
#
# Fix:
#   * Sort corregido: planos numericos primero, KP al final.
#   * Persistimos la KEY del plano seleccionado en session_state
#     y la usamos como `index=` del selectbox en cada rerun.
#     Asi el rerun sobrevive el cambio de label.
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add pages/00_Machinery_Library.py
git status --short | head

git commit -m "fix(ctp): dropdown vuelve al keyphasor despues de guardar (Ciclo 15.2)

Dos bugs juntos:
(1) Sort tenia signo invertido — KP quedaba primero (indice 0).
(2) st.rerun() perdia la seleccion porque la label cambia (de
'sin posicionar' a 'posicionado') y Streamlit no encuentra el
valor previo por label.

Fix: sort corregido (KP al final) + persistir la KEY del plano en
session_state y usarla como index en cada rerun." || echo "Nothing to commit"

git push origin dev

echo ""
echo "✓ Hotfix pusheado. Refrescá el browser."
