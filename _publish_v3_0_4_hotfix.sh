#!/bin/bash
# =============================================================
# Watermelon — v3.0.4 hotfix: compresor recip balanced-opposed
# =============================================================
# BUG REPORTADO:
# v3.0.3 dibujó el compresor con N cilindros TODOS HACIA ARRIBA.
# Pero ARIEL KBK/4 (y la mayoría de los recip industriales) son
# máquinas BALANCED-OPPOSED — los cilindros se distribuyen mitad
# arriba y mitad abajo del crankcase, opuestos en pares (los
# "throws" del cigüeñal). El usuario reportó: "solo el compresor
# presento 2 cilindros y son 4 cilindros, los otros dos van
# opuestos a los primero".
#
# FIX en core/sensor_diagram.py:
# Para driven_kind="recip_compressor", ahora distribuyo los N
# cilindros como n_throws = ceil(N/2) posiciones a lo largo del
# crankcase. En cada throw position van un cilindro UP y otro
# DOWN (cuando hay otro disponible). Asignación:
#   - cilindros pares (idx 0, 2, 4...) → UP
#   - cilindros impares (idx 1, 3, 5...) → DOWN
#   - throw_idx = i // 2 (mismo throw para par UP/DOWN)
#
# Resultado para ARIEL KBK/4 (N=4):
#   throw 1: cilindro 1 UP  + cilindro 2 DOWN (en mismo x)
#   throw 2: cilindro 3 UP  + cilindro 4 DOWN (en mismo x)
# Visualmente: 2 throws con par opuesto cada uno = 4 cilindros
# en configuración balanced-opposed. Coincide con el dibujo
# técnico real de un ARIEL KBK.
#
# Cada cilindro DOWN incluye su distance piece + cilindro +
# cabezal con válvulas (espejo del UP).
#
# También expandí ax_top.set_ylim a (-0.4, 4) cuando hay
# recip_compressor para que los cilindros DOWN no queden
# cortados por el borde inferior.
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.0.4"
RELEASE_TITLE="Hotfix: compresor reciprocante balanced-opposed (cilindros UP+DOWN)"

# ---------- Locks ----------
[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

# ---------- 0) Asegurar branch dev ----------
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# ---------- 1) Stage ----------
echo ""
echo "▶ Stageando cambios v3.0.4..."
git add core/sensor_diagram.py
git add _publish_v3_0_4_hotfix.sh 2>/dev/null || true

# ---------- 2) Commit ----------
if ! git diff --staged --quiet; then
    echo "▶ Commiteando hotfix v3.0.4..."
    git commit -m "fix(sensor_diagram): compresor reciprocante balanced-opposed (17.5.12)

Bug: v3.0.3 dibujaba todos los N cilindros del compresor
hacia ARRIBA. Pero ARIEL KBK/4 y la mayoria de recips
industriales son BALANCED-OPPOSED — mitad cilindros UP,
mitad DOWN, en pares opuestos. Usuario reporto: 'solo el
compresor presento 2 cilindros y son 4 cilindros, los otros
dos van opuestos a los primero'.

Fix: para driven_kind='recip_compressor', distribuyo N
cilindros como n_throws = ceil(N/2) posiciones a lo largo
del crankcase. Cilindros pares (idx 0,2,4...) van UP;
impares (idx 1,3,5...) van DOWN, opuestos al UP del mismo
throw (throw_idx = i // 2).

Para N=4 (ARIEL KBK/4):
  throw 1: cilindro 1 UP + cilindro 2 DOWN
  throw 2: cilindro 3 UP + cilindro 4 DOWN
Resultado visual: 2 throws con par opuesto cada uno =
configuracion balanced-opposed real.

Tambien expande ax_top.set_ylim a (-0.4, 4) cuando hay
recip_compressor para que los cilindros DOWN no queden
cortados por el borde inferior. Distance pieces + cabezales
con valvulas son espejo de los UP." || echo "  (sin cambios)"
else
    echo "  (no hay cambios staged)"
fi

# ---------- 3) Rebase ----------
echo ""
echo "▶ Reconciliando contra origin/dev..."
git fetch origin dev
git pull --rebase origin dev || {
    echo ""
    echo "✗ Rebase falló. Resolvelos a mano y re-ejecutá."
    exit 1
}

# ---------- 4) Push dev ----------
echo ""
echo "▶ Pusheando dev..."
git push origin dev

# ---------- 5) Switch a main ----------
echo ""
echo "▶ Cambiando a main..."
git checkout main
git pull origin main

# ---------- 6) Merge dev → main ----------
echo ""
echo "▶ Mergeando dev → main..."
git merge dev --no-ff -m "hotfix(${VERSION}): ${RELEASE_TITLE}

Sobre v3.0.3:

Compresor reciprocante ahora se dibuja en configuracion
balanced-opposed (mitad cilindros UP, mitad DOWN, en pares
opuestos por throw del cigueñal). ARIEL KBK/4 con 4 cilindros
ahora muestra 2 throws con par opuesto cada uno: cilindro 1+2
en throw 1, cilindro 3+4 en throw 2. Coincide con el dibujo
tecnico real del fabricante."

# ---------- 7) Tag ----------
echo ""
echo "▶ Creando tag ${VERSION}..."
git tag -a "${VERSION}" -m "${RELEASE_TITLE}

Hotfix sobre v3.0.3: compresores reciprocantes ahora se
dibujan en configuracion balanced-opposed (cilindros
distribuidos UP+DOWN en pares por throw). Refleja
correctamente la geometria fisica de maquinas como
ARIEL KBK series."

# ---------- 8) Push main + tag ----------
echo ""
echo "▶ Pusheando main + ${VERSION}..."
git push origin main
git push origin "${VERSION}"

# ---------- 9) Volver a dev ----------
echo ""
echo "▶ Volviendo a dev..."
git checkout dev

echo ""
echo "================================================================"
echo " ✓ ${VERSION} publicado a main"
echo "================================================================"
echo ""
echo " Streamlit Cloud auto-redeploya en ~2 min."
echo " ARIEL KBK/4 ahora dibuja: 2 cilindros UP + 2 DOWN"
echo " (4 cilindros en 2 throws balanced-opposed)."
echo "================================================================"
