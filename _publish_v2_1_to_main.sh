#!/bin/bash
# =============================================================
# Watermelon — v2.1 → MAIN: Ciclos 10A.4 / 12.x / 14a / 14b / 14b.2 / 14c
# =============================================================
# Hace TRES cosas en cadena:
#
# 1) Commitea el bug fix matcher (TRF ACELL → accel correcto) +
#    Ciclo 14c.2 (diagrama visual del Sensor Map) en dev.
# 2) Push a origin/dev.
# 3) Merge dev → main con --no-ff, tag v2.1 + v2.1-pre-main para
#    rollback de un solo comando.
# 4) Push main + tags.
# 5) Vuelve a dev para seguir trabajando sin conflictos.
#
# CICLOS QUE LLEGAN A MAIN EN ESTE MERGE:
#
# Reports (PDF clase mundial):
#   * 10A.4 — Tabla de Contenido automática en página 2
#
# Time Waveform Cat IV:
#   * 12   — 5 detectores + wrapper completo
#   * 12.1 — UX cleanup (auto-select + de-dup)
#   * 12.2 — Lenguaje cliente + fusión analytics
#
# Machinery Library cockpit (modelo activo-céntrico):
#   * 14a   — Library + auto-fill reportes + esquemático en Resumen
#             Ejecutivo
#   * 14b   — Wire del selector en Load Data + bloqueo si no hay activo
#   * 14b.2 — Tabular List auto-derivado de Library (criterion + alarm
#             + danger desde instancia)
#
# Sensor Map (per-instance):
#   * 14c.1   — Modelo de sensores + integración Tabular List + form
#               editable en Library
#   * 14c.1.1 — Generador estándar diferenciado driver/driven
#               (proximity_xy / axial_accel / accel_plus_velocity) +
#               keyphasor en coupling
#   * 14c.2   — Diagrama visual del mapa (vista lateral + polar R/L)
#               + bug fix matcher TRF ACELL
#
# 8+ hotfixes intermedios pulieron UX (selector cross-page, dropdown
# unidades, match permisivo, navegación NAV_ITEMS, etc.)
#
# COMPATIBILIDAD:
#  * Instancias creadas en v2.0 siguen siendo válidas (campos nuevos
#    con defaults vacíos, from_dict resiliente).
#  * Reports v2.0 sigue generando PDFs sin cambio si la instancia no
#    tiene esquemático ni sensores configurados.
#  * Compile clean en todos los archivos modificados.
#
# Ejecutar:
#   bash _publish_v2_1_to_main.sh
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

echo ""
echo "Estado del repo:"
git status --short | grep -vE '^\?\?' | head -10 || true

echo ""
echo "================================================================"
echo " STEP 1: Commit bug fix matcher + Ciclo 14c.2 en dev"
echo "================================================================"
git add core/sensor_map.py core/sensor_diagram.py pages/00_Machinery_Library.py
git status --short | head

git commit -m "feat(library): Ciclo 14c.2 — diagrama visual del Sensor Map + fix matcher TRF (dev)

(1) BUG FIX en core/sensor_map.resolve_sensor_for_point:
Cuando hay multiples candidates despues del filtro por type_hint, el
codigo previo devolvia None (fallback a global con familia incorrecta).
Resultado visible: TRF ACELL clasificado como Proximity en Tabular List.

Fix: tie-break inteligente en orden:
* substring del label industrial (1y_d, 2_rad_a)
* substring del plane_label ('TRF (LM6000)', 'CRF (LM6000)')
* tokens distintivos del csv_match_pattern (filtrando comunes
  acell/acc/x/y/vel/rad)
* fallback gracioso: si quedan multiples, devolver el primero del
  tipo correcto (mejor un match accel ambiguo que un fallback proximity
  incorrecto)

Smoke validado: 9 sensores TES1 (4 driver accel+velocity + 4 driven
proximity X-Y + 1 keyphasor) → todos los Points reales (TRF ACELL,
CRF ACELL, VE5807-VE5810, KPHGEN) matchean correctamente.

(2) NUEVO core/sensor_diagram.py:
* render_sensor_map_diagram(sensors) → PNG bytes
* Vista lateral del tren con bloques DRIVER y DRIVEN, cojinetes
  numerados (convencion API 670 / ISO 20816-1 driver→driven), keyphasor
  visible en coupling.
* Vista polar por plano con hemisferio L/R (mirando desde extremo del
  driver, 0° arriba), sondas marcadas en sus angulos fisicos con
  marker shape distinto por tipo (circulo proximity, cuadrado velocity,
  triangulo accel, estrella keyphasor).
* Leyenda con 4 tipos de sensor.
* Smoke: 87 KB PNG generado para layout TES1.

(3) Integracion en Machinery Library:
* Despues del preview-expander del mapa, nuevo bloque '🎯 Diagrama
  visual del mapa' que renderiza el PNG con st.image() + caption
  explicativo de la convencion API 670.
* Try/except defensivo: si matplotlib no esta disponible, omite limpio.

Compile clean. NO toca matematica de Tabular List ni del PDF report —
solo agrega visualizacion + corrige el bug del matcher."

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
PRE_MERGE_TAG="v2.1-pre-main-$(date +%Y%m%d-%H%M%S)"
git tag -a "$PRE_MERGE_TAG" origin/main -m "Snapshot de main antes del merge v2.1"
git push origin "$PRE_MERGE_TAG"
echo "  Tag de rollback creado: $PRE_MERGE_TAG"

echo ""
echo "================================================================"
echo " STEP 4: Checkout main, pull, merge --no-ff dev"
echo "================================================================"
git checkout main
git pull origin main

echo ""
echo "Merging dev → main..."
git merge --no-ff dev -m "merge: dev → main — Watermelon v2.1

Trae a main todos los ciclos desarrollados desde v2.0:

Reports clase mundial:
* 10A.4 — TOC automática en página 2 con dot leaders y bookmarks

Time Waveform Cat IV (módulo de waveforms al nivel rotordinámica):
* 12 — 5 detectores (AM Hilbert, asimetría, clipping, sawtooth, beating)
       + wrapper completo + classify_crest_factor 5 buckets
* 12.1 — UX (auto-select all + remove debug + de-dup metrics)
* 12.2 — Lenguaje cliente + fusión análisis automático

Machinery Library cockpit (modelo máquina-céntrico):
* 14a — Library promovida a página 0 + form editable extendido (8 tabs:
        Identificación, Tren, Operación, Soportes, Sondas, Setpoints,
        Mantenimiento, Esquemático) + auto-fill de Reports.meta + render
        del esquemático en Resumen Ejecutivo del PDF
* 14b — Wire del selector en Load Data + banner verde 'Cargando CSVs
        para: TES1' + bloqueo si no hay activo + etiquetado de signals
        con instance_id
* 14b.2 — Tabular List auto-derivado de Library (criterion + alarm/
          danger desde instancia, eliminación de inputs manuales del
          sidebar) + override avanzado en expander + helper
          core/tabular_defaults.py

Sensor Map per-instance (mapa físico de sensores API 670):
* 14c.1 — Modelo Instance.sensors + integración Tabular List
          (resolve_sensor_for_point con thresholds individuales) +
          form editable st.data_editor + helper core/sensor_map.py
* 14c.1.1 — Generador diferenciado driver/driven con 3 modos de
            instrumentación (proximity_xy / axial_accel /
            accel_plus_velocity) + keyphasor en coupling +
            tipo de sensor 'keyphasor' agregado al sistema
* 14c.2 — Diagrama visual del mapa (vista lateral del tren + polar
          por plano R/L) + bug fix matcher TRF ACELL

Compatibilidad: instancias v2.0 siguen siendo válidas (defaults vacíos
en campos nuevos, from_dict resiliente). Reports v2.0 sigue generando
PDFs sin cambio si no hay esquemático ni sensores configurados.

Compile clean. Smoke runtime validado en sandbox."

echo ""
echo "================================================================"
echo " STEP 5: Tag v2.1 y push main + tags"
echo "================================================================"
git tag -a "v2.1" -m "Watermelon v2.1 — Machinery Library cockpit + Sensor Map per-instance + TOC + Time Waveform Cat IV"
git push origin main
git push origin v2.1

echo ""
echo "================================================================"
echo " STEP 6: Volver a dev para seguir trabajando"
echo "================================================================"
git checkout dev

echo ""
echo "================================================================"
echo " ✓ MERGE A MAIN COMPLETADO — v2.1 LIVE"
echo "================================================================"
echo ""
echo "Tags creados:"
echo "  - $PRE_MERGE_TAG (snapshot de main pre-merge para rollback)"
echo "  - v2.1 (release tag)"
echo ""
echo "Ramas actualizadas:"
echo "  - origin/dev   = $(git rev-parse --short origin/dev)"
echo "  - origin/main  = $(git rev-parse --short origin/main)"
echo ""
echo "ROLLBACK RÁPIDO si algo se rompe en producción:"
echo "  git checkout main"
echo "  git reset --hard $PRE_MERGE_TAG"
echo "  git push origin main --force-with-lease"
echo ""
echo "Streamlit Cloud va a re-deployar automáticamente desde main."
echo "Esperar 1-2 min y validar el flujo end-to-end:"
echo ""
echo "  1. Login → Machinery Library aparece como página 0 ✓"
echo "  2. Activar TES1 (badge verde + esquemático embebido) ✓"
echo "  3. Sensor Map → 'Generar mapa estándar' configurable ✓"
echo "  4. Diagrama visual del mapa renderiza ✓"
echo "  5. Load Data → banner verde + bloqueo si falta activo ✓"
echo "  6. Tabular List → auto-derivado de instancia + sensores"
echo "     individuales (TRF ACELL ahora clasifica como Acceleration) ✓"
echo "  7. Time Waveforms → Cat IV diagnostics + auto-select all ✓"
echo "  8. Reports → TOC en página 2 + esquemático en Resumen"
echo "     Ejecutivo + train_description auto ✓"
echo ""
echo "================================================================"
