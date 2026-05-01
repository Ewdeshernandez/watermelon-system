#!/bin/bash
# =============================================================
# Watermelon — v3.0.1 hotfix (17.5.9): chip + planos 50
# =============================================================
# v3.0 ya está en main con todo hasta 17.5.8.
# Falta 17.5.9: chip health por max reciente + Machinery 50.
# Este script lo cierra.
#
# Diferencia con el v3.0 anterior: stagea ANTES de hacer pull
# rebase (era el bug que abortó el script previo cuando había
# cambios uncommitted en el working tree).
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.0.1"
RELEASE_TITLE="Hotfix 17.5.9: chip health por max reciente + Machinery 50 apoyos"

# ---------- Locks ----------
[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

# ---------- 0) Asegurar branch dev ----------
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# ---------- 1) STAGE primero (antes del rebase) ----------
echo ""
echo "▶ Stageando cambios 17.5.9..."
git add pages/04_Trends.py pages/00_Machinery_Library.py
git add _publish_v3_0_1_hotfix.sh 2>/dev/null || true

# ---------- 2) Commit (si hay cambios staged) ----------
if ! git diff --staged --quiet; then
    echo "▶ Commiteando hotfix 17.5.9..."
    git commit -m "fix(trend+library): chip health por max reciente + Machinery hasta 50 apoyos (17.5.9)

(1) BUG: Chip 'Normal' aunque hay datos sobre Alarm.
Antes _compute_trend_health clasificaba SOLO por latest_value
(ultimo sample). Si el ultimo punto estaba debajo de Warning
aunque la ventana tuviera multiples picos sobre Danger, decia
'Normal'. Caso reportado: pico 2.319 in/s pk con Danger 1.230
y latest 0.057 -> chip Normal.

Fix: clasificar por el WORST de la ventana reciente (ultimos
7 dias o ultimos 100 samples). Si cualquier punto reciente
cruza Danger -> action; cualquiera cruza Warning -> alarm.
Latest sigue siendo el numero reportado, status refleja el
peor reciente. Nuevo campo recent_max_value en health dict.
Prosa explicita el pico cuando latest << recent_max.

(2) BUG: Machinery Library no permitia mas de 16 apoyos.
gen_driver_planes y gen_driven_planes con max_value=8 cada
uno. Maquinas grandes (compresores multi-etapa, trenes con
multiples soportes) requerian mas. Subido a 50 por seccion
(100 max total) + columna plane del data editor a 100 +
support_count a 50."
else
    echo "  (no hay cambios staged — saltando commit)"
fi

# ---------- 3) Ahora SI rebase contra origin ----------
echo ""
echo "▶ Reconciliando contra origin/dev (rebase)..."
git fetch origin dev
git pull --rebase origin dev || {
    echo ""
    echo "✗ Rebase falló por conflictos. Resolvelos:"
    echo "    git status"
    echo "    git add <archivos>"
    echo "    git rebase --continue"
    echo "    bash _publish_v3_0_1_hotfix.sh"
    exit 1
}

# ---------- 4) Push dev ----------
echo ""
echo "▶ Pusheando dev a origin..."
git push origin dev

# ---------- 5) Switch a main + pull ----------
echo ""
echo "▶ Cambiando a main..."
git checkout main
git pull origin main

# ---------- 6) Merge dev → main ----------
echo ""
echo "▶ Mergeando dev → main..."
git merge dev --no-ff -m "hotfix(${VERSION}): ${RELEASE_TITLE}

Sobre v3.0:

(1) Chip 'Normal' cuando hay picos sobre Danger.
_compute_trend_health ahora clasifica el status por el peor
valor de la ventana reciente (ultimos 7 dias o ultimos 100
samples), no solo por el latest_value. Si cualquier punto
reciente cruza Warning/Danger, el chip escala correctamente.
La prosa del autodiag explicita el pico cuando latest <<
recent_max para que el lector no se confunda.

(2) Machinery Library hasta 50 apoyos por seccion.
gen_driver_planes y gen_driven_planes pasan de max_value=8
a max_value=50 (100 totales). Columna plane del data editor
a max_value=100. support_count a max_value=50."

# ---------- 7) Tag v3.0.1 ----------
echo ""
echo "▶ Creando tag ${VERSION}..."
git tag -a "${VERSION}" -m "${RELEASE_TITLE}

Hotfix sobre v3.0:

  (1) Chip health por max reciente: clasifica el status del
      Trend por el peor valor de la ventana reciente, no solo
      por el ultimo sample. El chip ahora escala a Atencion/
      Accion cuando hay picos recientes sobre Warning/Danger
      aunque el ultimo valor este abajo.

  (2) Machinery Library hasta 50 apoyos por seccion (100
      total entre driver+driven). Maquinas grandes con muchos
      soportes ahora se modelan completas en el Sensor Map."

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
echo " Branch actual: dev"
echo " Tag creado:    ${VERSION}"
echo ""
echo " Verificación:"
echo "   git log main --oneline | head -5"
echo "   git tag --list | grep v3.0"
echo "   git diff main dev --stat   # vacío = main contiene todo"
echo ""
echo " Después: refrescá Streamlit Cloud (auto-redeploya en ~2 min)"
echo " o reiniciá tu Streamlit local para ver los cambios."
echo "================================================================"
