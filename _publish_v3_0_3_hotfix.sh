#!/bin/bash
# =============================================================
# Watermelon — v3.0.3 hotfix: silhouettes adaptativos
# =============================================================
# BUG REPORTADO:
# 1) En Machinery Library el "Diagrama visual del mapa" siempre
#    dibuja una TURBINA + GENERADOR. Para máquinas que NO son
#    turbogenerador (motor eléctrico + compresor reciprocante,
#    bomba centrífuga, gearbox, etc.) el dibujo no representa
#    el activo real.
# 2) "Debe permitir 2 sensores en motor y resto en compresor".
#    Cubierto por 17.5.9 (driver_planes / driven_planes
#    independientes con max=50).
#
# FIX:
# core/sensor_diagram.py — render_sensor_map_diagram() acepta
# nuevos parámetros driver_kind y driven_kind con valores:
#   - turbine     (default driver: aero-derivative)
#   - motor       (rectangular + caja bornes + cooling fins)
#   - recip_compressor (crankcase + N cilindros verticales con
#                       cabezales de válvulas; N detectado de
#                       los planos cuyo plane_label contiene
#                       'cilindro')
#   - centrif_compressor (voluta tipo snail + descarga)
#   - pump        (voluta + descarga)
#   - gearbox     (caja con dos engranajes visibles)
#   - generator   (default driven: cilindro + end shields + fan vanes)
#
# Helper _infer_machine_kind(label) hace heurística por
# keywords (motor/turbine/recip/ariel/kbk/centrif/bomba/etc.)
# para autodetectar desde driver_label / driven_label /
# asset_class cuando el caller no pasa kind explícito.
#
# Las 3 llamadas a render_sensor_map_diagram en
# pages/00_Machinery_Library.py, pages/16_Reports.py (×2) y
# pages/01b_Machine_Map.py ahora infieren kind y lo pasan.
#
# Resultado para el caso del usuario (Compr recip C200C):
#   driver_label "HYUNDAI ELECTRIC HNP2 503-68E" -> motor
#   driven_label "ARIEL CORPORATION KBK/4"       -> recip_compressor
#   El diagrama muestra motor eléctrico + crankcase con 4
#   cilindros, no turbina + generador.
#
# Esta versión también recoge cualquier hotfix 17.5.9 que haya
# quedado uncommitted (chip health max reciente + planos 50).
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.0.3"
RELEASE_TITLE="Hotfix: silhouettes adaptativos motor/recip/centrif/pump/gearbox"

# ---------- Locks ----------
[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

# ---------- 0) Asegurar branch dev ----------
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# ---------- 1) Stage TODO lo modificado en pages/ y core/ ----------
echo ""
echo "▶ Stageando cambios pendientes en pages/ y core/..."
git add core/sensor_diagram.py
git add pages/00_Machinery_Library.py
git add pages/16_Reports.py
git add pages/01b_Machine_Map.py
git add pages/04_Trends.py 2>/dev/null || true   # por si 17.5.9 quedó uncommitted
git add _publish_v3_0_3_hotfix.sh 2>/dev/null || true

# ---------- 2) Commit ----------
if ! git diff --staged --quiet; then
    echo "▶ Commiteando hotfix v3.0.3..."
    git commit -m "fix(machinery): silhouettes adaptativos motor/recip/centrif/pump/gearbox (17.5.11)

Bug reportado: Diagrama visual del Mapa de Sensores siempre
dibujaba turbina + generador. Para activos que no son
turbogenerador (caso reportado: Compresor reciprocante C200C
con HYUNDAI ELECTRIC HNP2 503-68E motor + ARIEL CORPORATION
KBK/4 crankcase 4 cilindros) el dibujo no representaba el
activo real, lo que generaba reportes con esquematicos
incorrectos y bloqueaba la generacion de PDFs profesionales.

Fix en core/sensor_diagram.py:
  render_sensor_map_diagram acepta driver_kind y driven_kind:
    - turbine (default driver, aero-derivative)
    - motor (rectangular + caja bornes + cooling fins)
    - recip_compressor (crankcase + N cilindros con valvulas;
      N se detecta de los planos cuyo label contiene cilindro)
    - centrif_compressor (voluta snail + descarga)
    - pump (voluta + descarga)
    - gearbox (caja con engranajes visibles)
    - generator (default driven, cilindro + end shields + fan)

Helper _infer_machine_kind(label) autodetecta por keywords
(motor/turbine/recip/ariel/kbk/centrif/bomba/etc). Las 3
llamadas en Machinery Library, Reports (x2) y Machine Map
ahora pasan kind inferido del driver_label/driven_label/
asset_class.

Tambien recoge: 17.5.9 (chip health por max reciente +
Machinery max_value 50)." || echo "  (sin cambios)"
else
    echo "  (no hay cambios staged)"
fi

# ---------- 3) Rebase contra origin ----------
echo ""
echo "▶ Reconciliando contra origin/dev..."
git fetch origin dev
git pull --rebase origin dev || {
    echo ""
    echo "✗ Rebase falló. Resolvelos a mano:"
    echo "    git status"
    echo "    git add <archivos>"
    echo "    git rebase --continue"
    echo "    bash _publish_v3_0_3_hotfix.sh"
    exit 1
}

# ---------- 4) Push dev ----------
echo ""
echo "▶ Pusheando dev..."
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

Sobre v3.0.2:

Diagrama visual del Mapa de Sensores ahora soporta silhouettes
adaptativas segun el tipo de maquina (motor / recip /
centrifugal / pump / gearbox / turbine / generator).
_infer_machine_kind(label) autodetecta desde el driver_label /
driven_label / asset_class cuando el caller no pasa kind
explicito.

Caso del usuario (Compresor reciprocante kbk-4):
  driver HYUNDAI ELECTRIC HNP2 503-68E -> motor electrico
  driven ARIEL CORPORATION KBK/4 4 cilindros -> recip
  Diagrama muestra motor + crankcase con 4 cilindros
  verticales con cabezales, no turbina + generador.

Tambien incluye 17.5.9 si quedo uncommitted (chip health por
max reciente + Machinery 50 apoyos por seccion)."

# ---------- 7) Tag v3.0.3 ----------
echo ""
echo "▶ Creando tag ${VERSION}..."
git tag -a "${VERSION}" -m "${RELEASE_TITLE}

Hotfix sobre v3.0.2: silhouettes adaptativas para activos
que no son turbogenerador. El Mapa de Sensores ahora dibuja
motor electrico + crankcase con cilindros (compresor
reciprocante), voluta snail (centrifugo), voluta + descarga
(bomba), caja con engranajes (gearbox), o las defaults
turbine + generator. Heuristica de inferencia por keywords
en driver_label / driven_label / asset_class."

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
echo " Después refrescá Machinery Library (Cmd+Shift+R) y vas a ver:"
echo "   - C200C dibuja motor + crankcase con 4 cilindros"
echo "   - TES1 sigue dibujando turbina + generador"
echo "   - Wizard acepta 2 driver + 6 driven sin error"
echo "================================================================"
