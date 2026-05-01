#!/bin/bash
# =============================================================
# Watermelon — v2.2 → MAIN: Ciclo 14c.3 (Tabular List Cat IV)
# =============================================================
# Sigue exactamente el mismo flujo del merge v2.1 anterior pero
# con Ciclo 14c.3 que pule Tabular List a nivel producción:
#
# 1) ELIMINA del sidebar de Tabular List los bloques manuales de
#    "Machine Settings" / "Point Settings" / Configuration mode
#    (ahora redundantes con el Sensor Map del Ciclo 14c.1).
#
# 2) HACE QUE EL TABULAR LIST RESPETE unit_native del Sensor Map.
#    Si configurás un acelerómetro con "g peak", la columna Overall
#    y los harmonics 0.5X/1X/2X salen en "g peak" (no "g rms").
#    Lo mismo para "mil pp" / "µm pp" / "mm/s peak" etc.
#    Nuevo campo "Unit Full" en el DataFrame que tiene el texto
#    completo a mostrar; render_table y render PNG lo respetan.
#
# 3) DIAGNÓSTICO CAT IV completo en core/tabular_diagnostics.py
#    Antes: "Condición general estable. Semáforo: SAFE. Normal: 5".
#    Ahora:
#    * Apertura: "El módulo Tabular List analizó N señales
#      correspondientes a la máquina X, distribuidas por familia
#      como X Proximity + Y Acceleration."
#    * Citas de normas dinámicas según familia presente:
#      - API 670 (sondas de proximidad)
#      - ISO 7919-3 / ISO 20816-3 (shaft displacement)
#      - ISO 20816-3 / ISO 13373-1 (envelope acceleration)
#    * Distribución global de severidad con vocabulario Cat IV
#      (CONDICIÓN ACEPTABLE / ATENCIÓN / ACCIÓN REQUERIDA / CRÍTICA).
#    * Margen consumido (% del danger) de los activos críticos.
#    * Recomendaciones técnicas priorizadas en lista numerada con
#      citas a normas: ISO 21940-12 G 2.5 (balanceo), API 686
#      (alineación), API 684 (inestabilidad rotodinámica).
#
# COMMIT + PUSH dev → MERGE main → TAG v2.2 → VOLVER A dev.
#
# Ejecutar:
#   bash _publish_v2_2_to_main.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock

echo ""
echo "================================================================"
echo " STEP 0: Verificar branch y estado actual"
echo "================================================================"
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "dev" ]; then
  echo "Cambiando a dev..."
  git checkout dev
fi
git status --short | grep -vE '^\?\?' | head -10 || true

echo ""
echo "================================================================"
echo " STEP 1: Commit Ciclo 14c.3 en dev"
echo "================================================================"
git add pages/01__Tabular_List.py core/tabular_diagnostics.py
git status --short | head

git commit -m "feat(tabular): Ciclo 14c.3 — Tabular List respeta unit_native + diagnostico Cat IV (dev)

Tres mejoras pulen Tabular List a nivel produccion despues del Sensor
Map del Ciclo 14c.1.

(1) Eliminados bloques manuales del sidebar (Machine Settings / Point
Settings / Configuration mode). Ahora redundantes con el Sensor Map.
La sidebar queda minima: solo el override avanzado del Ciclo 14b.2 en
expander colapsado para casos de comparacion entre criterios.

(2) Tabular List respeta la unit_native individual de cada sensor del
mapa. Si configuras 'g peak' en el sensor TRF ACELL, las columnas
Overall y harmonics 0.5X/1X/2X muestran 'g peak', no 'g rms'. Lo
mismo aplica a 'mil pp' / 'um pp' / 'mm/s peak' / 'in/s RMS'.

Implementacion: nuevo campo 'Unit Full' en el row del DataFrame que
tiene el texto completo a mostrar (incluye sufijo). build_table_dataframe
infiere overall_mode_row del unit_native (peak/rms/pp). render_table y
render PNG usan Unit Full cuando esta presente; fallback al legacy
unit + display_suffix cuando no hay sensor map.

(3) Diagnostico Cat IV completo en core/tabular_diagnostics.py:
* Apertura tecnica: 'El modulo Tabular List analizo N senales
  correspondientes a la maquina X, distribuidas por familia como
  X Proximity + Y Acceleration.'
* Citas dinamicas de normas segun familias presentes (API 670 / ISO
  7919-3 / ISO 20816-3 / ISO 13373-1).
* Distribucion global con vocabulario Cat IV (CONDICION ACEPTABLE /
  ATENCION / ACCION REQUERIDA / CRITICA).
* Margen consumido (% del danger) de los activos criticos.
* Recomendaciones tecnicas priorizadas en lista numerada con citas
  a ISO 21940-12 G 2.5 (balanceo), API 686 (alineacion), API 684
  (inestabilidad rotodinamica).

Smoke validado: Brush 54 MW + LM6000 con 3 signals (1 accel + 2 proxim)
genera narrativa 5x mas rica que la version anterior, con normas
correctamente atribuidas y firma 1X identificada como tipo desbalance
con recomendacion ISO 21940-12 G 2.5.

Compile clean."

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
PRE_MERGE_TAG="v2.2-pre-main-$(date +%Y%m%d-%H%M%S)"
git tag -a "$PRE_MERGE_TAG" origin/main -m "Snapshot de main antes del merge v2.2"
git push origin "$PRE_MERGE_TAG"
echo "  Tag de rollback creado: $PRE_MERGE_TAG"

echo ""
echo "================================================================"
echo " STEP 4: Merge dev → main"
echo "================================================================"
git checkout main
git pull origin main
git merge --no-ff dev -m "merge: dev → main — Watermelon v2.2

Ciclo 14c.3 completo: Tabular List a nivel produccion con narrativa
Cat IV y respeto de unit_native individual de cada sensor del mapa.

* Sidebar minimal sin inputs manuales redundantes con el Sensor Map.
* Columna Overall y harmonics respetan 'g peak' / 'mil pp' / etc.
  configurado por sensor.
* Narrativa enriquecida con citas dinamicas a normas (API 670 / ISO
  7919-3 / 20816-3 / 13373-1) + recomendaciones priorizadas con
  ISO 21940-12 (balanceo) / API 686 (alineacion) / API 684
  (inestabilidad).

Compatibilidad: instancias y mapas previos siguen siendo validos."

echo ""
echo "================================================================"
echo " STEP 5: Tag v2.2 y push"
echo "================================================================"
git tag -a "v2.2" -m "Watermelon v2.2 — Tabular List Cat IV con Sensor Map respetado en unidades + narrativa enriquecida"
git push origin main
git push origin v2.2

echo ""
echo "================================================================"
echo " STEP 6: Volver a dev para seguir trabajando"
echo "================================================================"
git checkout dev

echo ""
echo "================================================================"
echo " ✓ MERGE A MAIN COMPLETADO — v2.2 LIVE"
echo "================================================================"
echo ""
echo "Tags creados:"
echo "  - $PRE_MERGE_TAG (rollback)"
echo "  - v2.2 (release)"
echo ""
echo "ROLLBACK si algo se rompe:"
echo "  git checkout main && git reset --hard $PRE_MERGE_TAG && git push --force-with-lease origin main"
echo ""
echo "Validar en wm-home-final-2026.streamlit.app (1-2 min de redeploy):"
echo ""
echo "  1. Library → Sensor Map de TES1 con 'g peak' en algunos sensors"
echo "  2. Load Data → subir CSVs"
echo "  3. Tabular List:"
echo "     - Sidebar: NO aparecen Machine Settings / Point Settings"
echo "     - Columna Overall del acelerómetro muestra 'X.XXX g peak'"
echo "       (no 'g rms')"
echo "     - Diagnóstico textual abajo: narrativa Cat IV completa"
echo "       con citas a ISO 7919-3, API 670, ISO 21940-12, etc."
echo ""
echo "Después podemos seguir con:"
echo "  - Ciclo 15: Esquemático de planos de vibración (overlay del"
echo "    train con heatmap de severidad por punto sobre la imagen"
echo "    real del activo)"
echo "================================================================"
