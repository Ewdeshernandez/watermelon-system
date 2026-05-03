#!/bin/bash
# =============================================================
# Watermelon — v3.0.5 hotfix: motor TEFC industrial profesional
# =============================================================
# Mejora estética del silhouette del motor eléctrico para que
# luzca como un TEFC induction motor industrial real
# (Totally Enclosed Fan Cooled, estándar IEC/NEMA).
#
# ANTES (v3.0.3):
#   - Caja rectangular plana
#   - Cooling fins solo en los primeros cm del lado outboard
#   - Caja de bornes pequeña (un rectángulo)
#   - End shields rounded sin detalles
#
# AHORA (v3.0.5):
#   ✓ Frame con cooling fins LONGITUDINALES recorriendo todo el
#     cuerpo (9 ribs horizontales evenly spaced)
#   ✓ Fan cover en el outboard end con GRILLA CIRCULAR cruzada
#     (4 brazos a 0/45/90/135°) + hub central
#   ✓ End shield del lado DE con bolt circle visible (4 tornillos)
#   ✓ Caja de bornes prominente con tapa (línea de partición) y
#     2 prensaestopas (cable glands) en la parte superior
#   ✓ Eje sobresaliendo del end shield DE hacia el coupling
#   ✓ Patas de montaje (mounting feet) abajo con tornillos de
#     anclaje visibles
#
# Resultado: el motor parece un motor industrial real, no una
# caja con fins. Funciona para cualquier motor eléctrico
# (HYUNDAI HNP2, WEG, ABB, Siemens, Toshiba, etc.).
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.0.5"
RELEASE_TITLE="Hotfix estético: motor TEFC industrial profesional"

# ---------- Locks ----------
[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

# ---------- 0) Asegurar branch dev ----------
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# ---------- 1) Stage ----------
echo ""
echo "▶ Stageando cambios v3.0.5..."
git add core/sensor_diagram.py
git add _publish_v3_0_5_hotfix.sh 2>/dev/null || true

# ---------- 2) Commit ----------
if ! git diff --staged --quiet; then
    echo "▶ Commiteando hotfix v3.0.5..."
    git commit -m "feat(sensor_diagram): motor electrico TEFC industrial profesional (17.5.13)

Mejora estetica del silhouette del motor (driver_kind='motor')
para que parezca un TEFC induction motor industrial real
(estandar IEC/NEMA), no una caja con fins.

Antes: caja rectangular plana + cooling fins solo en los
primeros cm del lado outboard + caja de bornes minima.

Ahora:
  - Frame con cooling fins LONGITUDINALES recorriendo todo el
    cuerpo (9 ribs horizontales evenly spaced)
  - Fan cover en el outboard end con grilla circular cruzada
    (4 brazos a 0/45/90/135) + hub central
  - End shield del lado DE con bolt circle visible (4 tornillos)
  - Caja de bornes prominente con tapa (linea de particion) +
    2 prensaestopas / cable glands en la parte superior
  - Eje sobresaliendo del end shield DE hacia el coupling
  - Patas de montaje (mounting feet) abajo con tornillos de
    anclaje visibles

Funciona para cualquier motor electrico industrial: HYUNDAI
HNP2, WEG, ABB, Siemens, Toshiba, etc." || echo "  (sin cambios)"
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

# ---------- 6) Merge ----------
echo ""
echo "▶ Mergeando dev → main..."
git merge dev --no-ff -m "hotfix(${VERSION}): ${RELEASE_TITLE}

Sobre v3.0.4:

Motor electrico ahora se dibuja como un TEFC induction motor
industrial profesional. Frame con cooling fins longitudinales
+ fan cover con grilla circular + caja de bornes con cable
glands + end shield con bolt circle + eje hacia coupling +
patas con tornillos de anclaje. Funciona para HYUNDAI HNP2,
WEG, ABB, Siemens y similares."

# ---------- 7) Tag ----------
echo ""
echo "▶ Creando tag ${VERSION}..."
git tag -a "${VERSION}" -m "${RELEASE_TITLE}

Mejora estetica del silhouette del motor electrico. Parece un
motor industrial real (TEFC IEC/NEMA) en lugar de una caja
con fins. Detalles: frame ribbed completo, fan cover con
grilla, caja bornes con glands, bolt circles, eje, patas."

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
echo " Refrescá Machinery Library con Cmd+Shift+R."
echo " El motor HYUNDAI HNP2 ahora se ve como un motor TEFC real:"
echo "   - Cooling fins horizontales en todo el cuerpo"
echo "   - Fan cover con grilla circular en el outboard"
echo "   - Caja de bornes prominente con cable entries"
echo "   - Eje saliendo hacia el coupling"
echo "   - Patas con tornillos de anclaje"
echo "================================================================"
