#!/bin/bash
# =============================================================
# Watermelon — v3.0.8 hotfix: compresor MINIMAL como pidió el usuario
# =============================================================
# Pedido textual del usuario: "haz una caja en el centro con un
# cigüeñal y arriba solo 2 cajas y abajo dos cajas más, arriba es
# cilindro 1 y 3 y abajo es cilindro 2 y 4".
#
# Diseño minimal, sin cabezales ni válvulas:
#   - Caja central horizontal con un círculo + cruz blanca como
#     símbolo del cigüeñal + etiqueta "cigüeñal" debajo
#   - 2 cajas chiquitas arriba: 1 (izquierda) y 3 (derecha)
#   - 2 cajas chiquitas abajo: 2 (izquierda) y 4 (derecha)
#   - Vástago/conector entre cada caja y el crankcase
#   - Número grande del cilindro centrado en cada caja
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.0.8"
RELEASE_TITLE="Hotfix: compresor minimal — caja con cigüeñal + 4 cajas etiquetadas"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

echo ""
echo "▶ Stageando v3.0.8..."
git add core/sensor_diagram.py
git add _publish_v3_0_8_hotfix.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commiteando v3.0.8..."
    git commit -m "fix(sensor_diagram): compresor minimal — caja central + 4 cajas (17.5.16)

Pedido textual del usuario: 'haz una caja en el centro con un
cigüeñal y arriba solo 2 cajas y abajo dos cajas más, arriba
es cilindro 1 y 3 y abajo es cilindro 2 y 4'.

Diseño explicitamente minimalista. Sin cabezales, sin valvulas,
sin cilindros horizontales. Solo:
  - Caja central horizontal con simbolo de cigueñal (circulo
    azul + cruz blanca) + etiqueta 'cigueñal' debajo
  - 2 cajas chiquitas arriba: '1' a la izquierda + '3' a la
    derecha
  - 2 cajas chiquitas abajo: '2' a la izquierda + '4' a la
    derecha
  - Vastago vertical conectando cada caja con el crankcase
  - Numero grande del cilindro en el centro de cada caja" || echo "  (sin cambios)"
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

Sobre v3.0.7:

Compresor reciprocante rediseñado al pedido textual del
usuario: caja central con cigueñal + 4 cajas chiquitas (1 y 3
arriba, 2 y 4 abajo), cada una numerada y conectada por
vastago al crankcase. Sin cabezales ni valvulas. Mucho mas
simple y reconocible."

echo "▶ Tag ${VERSION}..."
git tag -a "${VERSION}" -m "${RELEASE_TITLE}

Hotfix sobre v3.0.7: rediseño minimalista del compresor
reciprocante segun pedido textual del usuario. Caja central
con cigueñal + 4 cajas chiquitas (cilindros 1, 2, 3, 4)
arriba y abajo del crankcase."

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
echo " ARIEL KBK/4 ahora dibuja exactamente lo pedido:"
echo "   ┌──┐    ┌──┐"
echo "   │ 1│    │ 3│   ← cilindros 1 y 3 ARRIBA"
echo "   └─┬┘    └─┬┘"
echo "     │       │"
echo "   ╔═┴═══════┴═╗"
echo "   ║   ⊕ cig   ║   ← caja central con cigüeñal"
echo "   ╚═┬═══════┬═╝"
echo "     │       │"
echo "   ┌─┴┐    ┌─┴┐"
echo "   │ 2│    │ 4│   ← cilindros 2 y 4 ABAJO"
echo "   └──┘    └──┘"
echo "================================================================"
