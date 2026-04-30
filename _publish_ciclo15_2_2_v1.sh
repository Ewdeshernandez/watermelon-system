#!/bin/bash
# =============================================================
# Watermelon — Ciclo 15.2.2 → DEV: aire en stack de valores
# =============================================================
# Mejora visual: mas espacio vertical entre valores stackeados
# bajo cada cojinete + fonts un poco mas grandes para legibilidad
# sobre la foto real del activo.
#
#   - marker_radius           14 -> 16 px (cojinetes mas robustos)
#   - label_font (plane)      W/90  -> W/80 (mas grande)
#   - value_font (overall)    W/100 -> W/88 (mas grande)
#   - num_font (numero coj)   W/80  -> W/70 (mas grande)
#   - gap entre lineas valor  4 -> 10 px
#   - gap label plano-valor   6 -> 12 px
#   - padding fondo blanco    3 -> 6 px (x), 1 -> 4 px (y)
#   - alpha fondo blanco      220 -> 235 (mas opaco, legible
#                                          sobre cualquier foto)
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add core/sensor_diagram.py
git status --short | head

git commit -m "feat(diagram): aire entre valores stackeados sobre esquematico (Ciclo 15.2.2)

Mas espacio vertical entre lineas (4 -> 10 px), mas padding y
opacidad en el fondo blanco de cada label, fonts un poco mas
grandes. Asi el stack 3X_D / 3Y_D no queda pegado y se lee
comodo sobre la foto real del activo." || echo "Nothing to commit"

git push origin dev

echo ""
echo "✓ Pusheado. Refrescá el browser."
