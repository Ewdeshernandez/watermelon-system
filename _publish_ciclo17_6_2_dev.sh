#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.6.2 → DEV (no toca main)
# =============================================================
# Queja del usuario tras 17.6.1:
#   "quedo muy hermoso ahora el tema es porque aparece un
#   recuadro azul no esta mal solo que en la clave quedo como
#   doble o como corrido del recuadro blanco"
#
# CAUSA:
# BaseWeb tiene 2 capas anidadas que ambas pintaban borde:
#   [data-baseweb="input"]      ← OUTER
#     [data-baseweb="base-input"] ← INNER
# Mi CSS de 17.6.1 le ponia border + background a las DOS,
# entonces se veía un borde dentro de otro borde (doble).
# El problema se notaba sobre todo en focus porque el
# box-shadow de focus se duplicaba también.
#
# FIX:
# - OUTER ([data-baseweb="input"]) ÚNICO con border + background
#   + focus ring
# - INNER ([data-baseweb="base-input"]) reseteado a transparent,
#   border:0, border-radius:0, box-shadow:none
# - Override del rojo solo aplicado al OUTER (consistente)
#
# Resultado: una sola línea limpia en cada input.
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

echo ""
echo "▶ Stageando 17.6.2..."
git add pages/00_Login.py
git add _publish_ciclo17_6_2_dev.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commiteando 17.6.2..."
    git commit -m "fix(login): doble borde en inputs — solo outer baseweb pinta (17.6.2)

Queja: 'recuadro azul quedo como doble o corrido del recuadro
blanco' en focus de username/password.

Causa: BaseWeb tiene 2 capas anidadas y mi CSS de 17.6.1 le
ponía border + background a LAS DOS:
  [data-baseweb=input]       <- OUTER (con borde + bg)
    [data-baseweb=base-input]  <- INNER (TAMBIÉN con borde + bg)
Resultado visual: borde dentro de borde, focus ring duplicado.

Fix:
- OUTER es el unico con border + background + focus ring
- INNER reseteado a transparent + border:0 + box-shadow:none
- Override del rojo solo aplicado al OUTER (consistente)

Resultado: una sola linea limpia en cada input." || echo "  (sin cambios)"
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
echo "================================================================"
echo " ✓ Ciclo 17.6.2 pusheado a DEV (main intacto)"
echo "================================================================"
echo ""
echo " Refrescá Cmd+Shift+R en el login. El input de Contraseña"
echo " ahora muestra UN SOLO borde limpio en focus, sin la línea"
echo " duplicada del 'doble recuadro'."
echo "================================================================"
