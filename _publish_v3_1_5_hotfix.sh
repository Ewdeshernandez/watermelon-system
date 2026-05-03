#!/bin/bash
# =============================================================
# Watermelon — v3.1.5 hotfix: HD export ISO strings + diagnostic
# =============================================================
# Después de 4 intentos previos (v3.1.1-1.4) que NO resolvieron
# el bug del PNG HD vacío, mi mejor hipótesis ahora:
#
# CAUSA SOSPECHADA:
# Kaleido (PNG renderer) NO interpreta bien los `datetime64[ns]`
# que vienen en los traces vía to_dict(). Probablemente los
# serializa como int64 nanosegundos crudos, pero el x-axis
# tiene range en formato datetime — los puntos quedan FUERA del
# rango visible y los traces no se dibujan.
#
# FIX:
# 1. Convertir EXPLÍCITAMENTE las x de cada trace a strings ISO
#    "%Y-%m-%d %H:%M:%S" ANTES de pasar a kaleido. ISO strings
#    son universalmente entendibles por cualquier renderer.
# 2. Forzar autorange=True en xaxis para que kaleido recompute
#    el rango usando la data ISO nueva (no el range datetime
#    cacheado del original).
# 3. Engine="kaleido" explícito.
# 4. Validar que kaleido devuelva bytes >1000. Si no, devolver
#    error VISIBLE al usuario en lugar de None silencioso.
# 5. Si no hay traces con data, devolver mensaje explícito
#    "verificá Signal Selection".
#
# Si esto NO funciona, el siguiente paso sería downgradear
# kaleido (1.0.0 → 0.2.1) o cambiar a engine="orca", pero antes
# necesito ver el mensaje de error visible.
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.1.5"
RELEASE_TITLE="Hotfix: HD export ISO strings + diagnóstico explícito"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

echo "▶ Stageando..."
git add pages/04_Trends.py
git add _publish_v3_1_5_hotfix.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commit..."
    git commit -m "fix(trend): HD export ISO strings + autorange + error visible (17.8.5)

Tras 4 intentos previos que no resolvieron el PNG HD vacio,
mi mejor hipotesis: kaleido no interpreta bien datetime64[ns]
en traces serializados via to_dict() — los pasa como int64 ns
crudos pero el x-axis range esta en datetime, los puntos
quedan fuera del rango visible.

Fix:
1. Convertir x de cada trace a ISO strings antes de kaleido
2. autorange=True en xaxis para recomputar con data ISO
3. engine='kaleido' explicito
4. Validar bytes >1000, si no devolver error visible
5. Si no hay traces con data, mensaje explicito en lugar
   de PNG vacio silencioso

Si esto no funciona, siguiente paso es downgradear kaleido o
cambiar a engine='orca'." || echo "  (sin cambios)"
else
    echo "  (no hay cambios)"
fi

echo "▶ Reconciliando..."
git fetch origin dev
git pull --rebase origin dev || { echo "✗ Rebase falló."; exit 1; }

echo "▶ Push dev..."
git push origin dev

echo "▶ Switch a main..."
git checkout main
git pull origin main

echo "▶ Merge dev → main..."
git merge dev --no-ff -m "hotfix(${VERSION}): ${RELEASE_TITLE}

Sobre v3.1.4 (que tampoco resolvio):

HD export ahora convierte explicitamente datetime a ISO
strings antes de pasar a kaleido. Si todavia falla, el error
sera VISIBLE al usuario en lugar de PNG vacio silencioso —
para que podamos diagnosticar la causa real."

echo "▶ Tag ${VERSION}..."
git tag -a "${VERSION}" -m "${RELEASE_TITLE}

Hotfix sobre v3.1.4. Convierte timestamps a ISO strings antes
de kaleido. Reporta error visible si falla."

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
echo " Refrescá Cmd+Shift+R y probá Prepare PNG HD."
echo ""
echo " IMPORTANTE: si SIGUE saliendo vacío, ahora vas a ver un"
echo " mensaje de error VISIBLE explicando qué pasó (en lugar"
echo " del PNG vacío silencioso). Mandame el mensaje y resolvemos."
echo "================================================================"
