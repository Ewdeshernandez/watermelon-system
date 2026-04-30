#!/bin/bash
# =============================================================
# Watermelon — v2.3 → MAIN: Sensor Map robusto + Machine Map
# =============================================================
# Cierra todos los hotfixes acumulados desde v2.2 y los lleva a main:
#
# 1) Ciclo 15.1: Machine Map nueva pagina (heatmap de severidad
#    por sensor en el tren acoplado, drill-down de criticos).
#
# 2) Hotfix Bently VT: matcher detecta velocity transducer Bently
#    por substring del Point name (1VT/2VT/VT/VEL) ANTES que por
#    unit (que en VT Bently puede reportarse como mil pp).
#
# 3) Fix critico de pre-filtro por type_hint: el pattern matcher
#    AHORA pre-filtra el universo de candidates por type_hint del
#    Point name antes de hacer pattern matching. Esto elimina
#    falsos matches cross-tipo donde por ejemplo el pattern
#    proximity '*4*x*' matcheaba contra la variable 'Vel Wf(64X/
#    32revs).KPHGEN' porque '4x' es substring de '64x'.
#    Adicionalmente: pattern matching solo chequea point_norm,
#    no variable_norm.
#
# 4) Cleanup del banner debug: quitado el banner amarillo
#    BUILD 14c.3-debug-v2 y el expander de debug ahora colapsado
#    por default. La página queda limpia para producción.
#
# COMMIT + PUSH dev → MERGE main → TAG v2.3 → VOLVER A dev.
#
# Ejecutar:
#   bash _publish_v2_3_to_main.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " STEP 0: Verificar branch y estado"
echo "================================================================"
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "dev" ]; then
  git checkout dev
fi
git status --short | grep -vE '^\?\?' | head

echo ""
echo "================================================================"
echo " STEP 1: Commit cleanup + fix final pre-filtro en dev"
echo "================================================================"
git add core/sensor_map.py pages/01__Tabular_List.py 2>/dev/null || true

git commit -m "fix(sensor-map): pre-filtro type_hint elimina falsos matches + cleanup debug

(1) FIX CRITICO de matcher con datos reales Bently:
Pattern '*4*x*' del proximity 4X_D matcheaba contra variable
'Vel Wf(64X/32revs).KPHGEN' porque '4x' es substring de '64x'
(oversampling rate del Bently). Velocímetros 1VT6805/1VT6831
caian como Family=Proximity con mil pp en lugar de Velocity
con in/s peak.

Fix: pre-filtrar candidates por type_hint detectado del Point
name (1VT/2VT/VT → velocity) ANTES del pattern matching. Si
type_hint esta claro, los patterns de otros tipos NUNCA se
evaluan. Adicionalmente: pattern matching solo chequea
point_norm (no variable_norm que tenia metadata tecnica
confusa).

Smoke validado contra Sensor Map REAL del usuario y CSVs reales:
* 1VT6805 (C) TRF → 2_RAD_V (velocity TRF) A=1.5/D=2.0 in/s peak ✓
* 1VT6831 (C) CRF → 1_RAD_V (velocity CRF) A=1.5/D=2.0 in/s peak ✓
* CRF/TRF ACELL → accelerometer A=3.0/D=6.0 g peak ✓
* VE5807-5810 → proximity ✓
* KPHGEN → keyphasor ✓

(2) Cleanup pages/01__Tabular_List.py: removido banner amarillo
'BUILD 14c.3-debug-v2' y expander de debug pasa a colapsado por
default. La pagina queda limpia para produccion." || echo "Nothing to commit"

echo ""
echo "================================================================"
echo " STEP 2: Push a origin/dev"
echo "================================================================"
git push origin dev

echo ""
echo "================================================================"
echo " STEP 3: Tag pre-merge en main para rollback"
echo "================================================================"
git fetch origin
PRE_MERGE_TAG="v2.3-pre-main-$(date +%Y%m%d-%H%M%S)"
git tag -a "$PRE_MERGE_TAG" origin/main -m "Snapshot de main antes del merge v2.3"
git push origin "$PRE_MERGE_TAG"
echo "  Tag de rollback creado: $PRE_MERGE_TAG"

echo ""
echo "================================================================"
echo " STEP 4: Merge dev → main"
echo "================================================================"
git checkout main
git pull origin main
git merge --no-ff dev -m "merge: dev → main — Watermelon v2.3

Trae a main todos los avances desde v2.2:

Machine Map (Ciclo 15.1):
* Nueva pagina con heatmap de severidad por sensor sobre el
  diagrama del tren acoplado (vista lateral + polar por plano).
* Tabla drill-down de sensores con atencion requerida.
* Integrado en menu lateral entre Tabular List y Time Waveforms.

Sensor Map matcher robusto (8+ hotfixes acumulados):
* Detección VT Bently (Velocity Transducers) por substring del
  Point name antes que por unit.
* Tie-break por tokens cortos del plane_label (TRF/CRF/BRG).
* Pre-filtro por type_hint antes de pattern matching: elimina
  falsos matches cross-tipo donde un pattern proximity '*4*x*'
  matcheaba contra metadata tecnica de la variable.
* Generador estandar de mapas con patterns separados por tipo.
* Fix indentacion machine_settings/point_settings undefined sin
  override.

Cleanup:
* Banner debug 'BUILD 14c.3-debug-v2' removido.
* Expander de debug del matcher pasa a colapsado por default.

Smoke validado end-to-end con datos reales del usuario (Brush
54 MW + LM6000, CSVs Bently con VT en in/s y acelerometros en
g peak)."

echo ""
echo "================================================================"
echo " STEP 5: Tag v2.3 y push"
echo "================================================================"
git tag -a "v2.3" -m "Watermelon v2.3 — Machine Map + Sensor Map matcher robusto Bently VT"
git push origin main
git push origin v2.3

echo ""
echo "================================================================"
echo " STEP 6: Volver a dev"
echo "================================================================"
git checkout dev

echo ""
echo "================================================================"
echo " ✓ MERGE A MAIN COMPLETADO — v2.3 LIVE"
echo "================================================================"
echo ""
echo "Tags creados:"
echo "  - $PRE_MERGE_TAG (rollback)"
echo "  - v2.3 (release)"
echo ""
echo "ROLLBACK:"
echo "  git checkout main && git reset --hard $PRE_MERGE_TAG && git push --force-with-lease origin main"
echo ""
echo "Streamlit Cloud va a redeployar en 1-2 min desde main."
echo ""
echo "Próximos pasos sugeridos:"
echo "  - Ciclo 15.1.1: integrar mini Machine Map arriba de Tabular List"
echo "  - Ciclo 15.1.2: integrar Machine Map en PDF report despues del"
echo "    Resumen Ejecutivo"
echo "  - Ciclo 15.2 (futuro): coordenadas 2D explicitas sobre la"
echo "    imagen real del activo (click-to-place)"
echo "================================================================"
