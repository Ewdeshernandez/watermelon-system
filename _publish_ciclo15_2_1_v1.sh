#!/bin/bash
# =============================================================
# Watermelon — Ciclo 15.2.1 → DEV: render multi-sensor por plano
# =============================================================
# Antes el render mostraba solo el sensor con mayor margen
# consumido del Danger por plano. Ahora muestra TODOS los
# sensores del plano con sus valores Overall individuales,
# cada uno coloreado por SU propia severidad. Asi:
#
#  - En TRF/CRF se ven el VELOCITY transducer Y el
#    ACELEROMETRO con sus valores propios.
#  - En los planos del generador (DE/NDE) se ven las sondas
#    X y Y por separado con sus mil pp respectivos.
#
# Aplica tanto al render_on_schematic (foto real del activo)
# como al render generico turbomachinery silhouette en modo
# compact. Los valores se ordenan por % de Danger consumido
# descendente — el sensor mas critico aparece primero.
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

git commit -m "feat(diagram): mostrar TODOS los sensores por plano (Ciclo 15.2.1)

Antes solo mostraba el peor sensor por plano. Ahora cada plano
lista todos sus sensores con valores Overall individuales y
coloreado por severidad propia. En TRF/CRF se ven vel + accel,
en planos del generador se ven X y Y. Aplica al render_on_schematic
(foto real) y al render compact generico." || echo "Nothing to commit"

git push origin dev

echo ""
echo "✓ Pusheado. Refrescá el browser."
