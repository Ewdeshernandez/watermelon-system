#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.1.2 UX → DEV: indicador legacy snapshots
# =============================================================
# Mejora UX para que sea OBVIO cuales snapshots Polar tienen
# trayectoria completa (post Ciclo 17.1.2) y cuales son legacy
# (solo operating point — los que existian antes).
#
# Cambios pages/06_Polar_Plot.py:
#
# (1) Warning visible en sidebar cuando hay snapshots legacy:
#     "⚠️ N snapshot(s) viejos sin trayectoria completa — solo
#     muestran el operating point en el polar. Para ver el loop
#     completo, resnapshoteá cargando esa corrida y volviendo a
#     guardar."
#
# (2) Boton 'Borrar los N snapshot(s) sin trayectoria' en el
#     expander Gestionar snapshots Polar — borra solo los legacy
#     en bloque, sin tocar los actuales.
#
# (3) En la lista de snapshots, chip por snapshot:
#     - 🟢 con trayectoria  (loop completo se dibuja)
#     - 🟡 solo punto Op (legacy) (solo estrella aparece)
#
# Ejecutar:
#   bash _publish_ciclo17_1_2_ux_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add pages/06_Polar_Plot.py
git status --short | head

git commit -m "feat(polar): UX indicador trayectoria + bulk delete legacy (17.1.2 UX)

Hace OBVIO en el sidebar cuales snapshots Polar tienen trayectoria
completa (post 17.1.2) y cuales son legacy (solo Op point — los
que existian antes del cambio).

Warning visible al inicio + boton bulk-delete de legacy + chip
indicador por snapshot en la lista (verde con trayectoria,
amarillo solo punto Op)." || echo "Nothing to commit"

git push origin dev

echo ""
echo "✓ UX pusheada a dev. Refrescá."
echo ""
echo "El sidebar ahora muestra:"
echo "  - Warning amarillo arriba con cuántos snapshots viejos hay"
echo "  - Botón 'Borrar los N sin trayectoria' (en Gestionar)"
echo "  - Chip por snapshot: 🟢 con trayectoria / 🟡 solo Op legacy"
