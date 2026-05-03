#!/bin/bash
# =============================================================
# Watermelon — v3.0.7 hotfix: cilindros HORIZONTALES (ARIEL real)
# =============================================================
# Usuario subió foto del esquemático real del ARIEL KBK/4 y
# explicó: "hay un cigüeñal en el centro, salen los 4 cilindros,
# vea el tuyo solo salen 2 cilindros".
#
# El usuario tiene razón — los compresores ARIEL KBK son
# HORIZONTAL-OPPOSED, no vertical-opposed:
#   - Crankcase: bloque central VERTICAL (más alto que ancho)
#   - Cigüeñal: en el centro horizontal del crankcase (eje
#     perpendicular al rotor)
#   - Cilindros: salen HORIZONTALMENTE hacia los lados
#     (2 al lado izquierdo, 2 al lado derecho del crankcase)
#
# REDISEÑO COMPLETO del recip_compressor:
#
# (1) Crankcase ahora es un bloque CENTRAL vertical (1.55 alto,
#     32% del ancho del driven section). Antes era horizontal
#     (full width, alto bajo).
#
# (2) Cigüeñal visible en el centro del crankcase: círculo
#     grande con cruz interior (representa el rotor) +
#     etiqueta "cigüeñal" debajo.
#
# (3) Cilindros horizontales extendiéndose hacia los lados:
#     - N=4 (ARIEL KBK/4): 2 LEFT (top + bottom) + 2 RIGHT
#       (top + bottom). Coincide con el dibujo real.
#     - N=2: 1 LEFT mid + 1 RIGHT mid
#     - N=6: 3 LEFT (top/mid/bot) + 3 RIGHT (top/mid/bot)
#     - Otros números: distribución alternada.
#
# (4) Cada cilindro tiene:
#     - Cuerpo horizontal (rectángulo rounded)
#     - Cabezal con válvulas en el EXTREMO LEJANO al crankcase
#     - 4 stubs de válvulas en el cabezal (succión/descarga
#       arriba y abajo)
#     - Número del cilindro (1, 2, 3, 4) en el centro del cuerpo
#
# (5) Largo del cilindro calculado para que el cabezal del
#     LEFT side no choque con el coupling (margen 0.22 desde
#     dvn_left).
#
# (6) ylim revertido a (0, 4.2) — los cilindros horizontales
#     no bajan más allá de y≈1.0, no necesitamos negativo.
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.0.7"
RELEASE_TITLE="Hotfix: compresor reciprocante con cilindros HORIZONTALES (ARIEL real)"

# ---------- Locks ----------
[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

# ---------- 0) Asegurar branch dev ----------
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# ---------- 1) Stage ----------
echo ""
echo "▶ Stageando v3.0.7..."
git add core/sensor_diagram.py
git add _publish_v3_0_7_hotfix.sh 2>/dev/null || true

# ---------- 2) Commit ----------
if ! git diff --staged --quiet; then
    echo "▶ Commiteando v3.0.7..."
    git commit -m "fix(sensor_diagram): cilindros recip horizontales como ARIEL KBK real (17.5.15)

Usuario subio foto del esquematico real ARIEL KBK/4 y explico:
'hay un cigueñal en el centro, salen los 4 cilindros, el tuyo
solo salen 2 cilindros'.

Tenia razon — los compresores ARIEL KBK son HORIZONTAL-OPPOSED:
crankcase central VERTICAL con cigueñal horizontal, y cilindros
saliendo HORIZONTALMENTE hacia los lados (no vertical-opposed
UP/DOWN como yo los dibujaba).

Rediseño completo del bloque recip_compressor:

(1) Crankcase: bloque central VERTICAL (1.55 alto, 32% del
    ancho del driven). Antes era horizontal full-width.

(2) Cigueñal visible en el centro: circulo + cruz interior +
    etiqueta 'cigueñal'.

(3) Cilindros horizontales extendiendose hacia los lados:
    N=4 -> 2 LEFT (top/bot) + 2 RIGHT (top/bot)
    N=2 -> 1 LEFT mid + 1 RIGHT mid
    N=6 -> 3 LEFT + 3 RIGHT
    Otros: alternados.

(4) Cada cilindro: cuerpo horizontal + cabezal en extremo
    lejano + 4 valvulas en cabezal + numero (1,2,3,4) en
    el centro del cuerpo.

(5) Largo calculado para que el cabezal LEFT no choque con
    el coupling (margen 0.22 desde dvn_left).

(6) ylim revertido a (0, 4.2)." || echo "  (sin cambios)"
else
    echo "  (no hay cambios staged)"
fi

# ---------- 3) Rebase ----------
echo ""
echo "▶ Reconciliando contra origin/dev..."
git fetch origin dev
git pull --rebase origin dev || { echo "✗ Rebase falló."; exit 1; }

# ---------- 4) Push dev ----------
echo "▶ Pusheando dev..."
git push origin dev

# ---------- 5) Switch a main ----------
echo "▶ Cambiando a main..."
git checkout main
git pull origin main

# ---------- 6) Merge ----------
echo "▶ Mergeando dev → main..."
git merge dev --no-ff -m "hotfix(${VERSION}): ${RELEASE_TITLE}

Sobre v3.0.6:

Compresor reciprocante ahora se dibuja en configuracion
HORIZONTAL-OPPOSED (estilo ARIEL KBK real), no vertical-opposed.
Crankcase central vertical con cigueñal visible + cilindros
horizontales saliendo hacia los lados (2 LEFT + 2 RIGHT para
N=4). Coincide con el dibujo tecnico real del usuario."

# ---------- 7) Tag ----------
echo "▶ Creando tag ${VERSION}..."
git tag -a "${VERSION}" -m "${RELEASE_TITLE}

Hotfix sobre v3.0.6: compresores reciprocantes en config
horizontal-opposed real. ARIEL KBK/4 ahora dibuja crankcase
central + 4 cilindros horizontales (2 hacia la izquierda + 2
hacia la derecha) con cabezales y valvulas en el extremo
lejano. Coincide con foto del fabricante."

# ---------- 8) Push main + tag ----------
echo "▶ Pusheando main + ${VERSION}..."
git push origin main
git push origin "${VERSION}"

# ---------- 9) Volver a dev ----------
echo "▶ Volviendo a dev..."
git checkout dev

echo ""
echo "================================================================"
echo " ✓ ${VERSION} publicado a main"
echo "================================================================"
echo ""
echo " ARIEL KBK/4 ahora se ve como el dibujo real:"
echo "   - Crankcase central con cigueñal"
echo "   - 2 cilindros horizontales LEFT (top + bottom)"
echo "   - 2 cilindros horizontales RIGHT (top + bottom)"
echo "   - Numerados 1, 2, 3, 4 con cabezales y valvulas"
echo "================================================================"
