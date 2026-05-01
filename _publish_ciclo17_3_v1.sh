#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.3 → DEV: SCL history (P1+P2+P3)
# =============================================================
# Snapshot del Shaft Centerline por bearing pair X-Y con:
#   - X/Y position del muñón a velocidad operativa (mil)
#   - Eccentricity ratio (0-1) y attitude angle (deg) calculados
#     vía core.scl_diagnostics.compute_eccentricity_state
#   - Lift-off speed detectado por heurística (eccentricity < 0.95)
#   - Trayectoria completa speed/x_gap/y_gap downsampleada a 80
#     puntos
#
# Permite ver entre corridas:
#   - Migración del centerline (X/Y position cambió)
#   - Cambio de eccentricity (carga / aceite / clearance)
#   - Shift de attitude angle (distribución de carga)
#   - Lift-off speed evolution (degradación hidrodinámica)
#
# Cambios:
#
# (P1) core/scl_history.py NUEVO — paralelo a polar/bode_history,
#     archivos scl_*.json. save/list/load/delete +
#     get_scl_history_for_bearing + get_previous_scl_snapshot con
#     skip_identical (tolerancia 0.05 mil en X/Y, 0.02 en
#     eccentricity ratio).
#
#     Clasificadores diagnósticos específicos del SCL:
#       eccentricity_change_classifier:
#         <0.05 stable
#         0.05-0.15 migration_minor
#         0.15-0.25 migration_major
#         >=0.25 migration_critical
#       attitude_shift_classifier:
#         <5° stable
#         5-15° shift_minor
#         15-30° shift_major
#         >=30° shift_critical
#
# (P2) pages/09_Shaft_Centerline.py — sidebar 'Histórico SCL' con
#     multiselect + extracción usando compute_eccentricity_state +
#     tabla comparativa inline con bearing label, X/Y position,
#     eccentricity, attitude angle entre corridas + diagnóstico.
#
#     Cuando los CSVs SCL no matchean el Sensor Map: expander de
#     diagnóstico mostrando Point/Paired Point Names del CSV vs
#     patterns del mapa (mismo patrón que Bode).
#
# (P3) _build_scl_report_notes acepta nueva clave
#     'comparison_narrative' en text_diag y la inserta entre
#     detalle y acciones. Pendiente para próxima iteración:
#     inyectar narrativa modal completa estilo Bently/API 670 para
#     SCL desde render_scl_panel (usando los clasificadores de
#     scl_history).
#
# Diferido para próxima iteración (P4):
#   * Overlay visual de trayectorias x_gap/y_gap históricas
#     sobre el SCL plot. Requiere modificar render_scl_panel
#     (~600 líneas). Por ahora la tabla comparativa muestra los
#     deltas numéricos.
#
# Ejecutar:
#   bash _publish_ciclo17_3_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add core/scl_history.py
git add pages/09_Shaft_Centerline.py
git status --short | head

git commit -m "feat(scl): SCL history multi-snapshot + comparativo (Ciclo 17.3 P1+P2+P3)

Snapshot del Shaft Centerline por bearing con X/Y position +
eccentricity ratio + attitude angle + lift-off speed +
trayectoria completa speed/x_gap/y_gap downsampleada.

(P1) core/scl_history.py NUEVO — paralelo a polar/bode_history.
save/list/load/delete + history_for_bearing + get_previous con
skip_identical. Clasificadores eccentricity_change_classifier
(stable/migration_minor/major/critical por threshold 0.05/0.15/
0.25) y attitude_shift_classifier (stable/shift_minor/major/
critical por threshold 5°/15°/30°).

(P2) pages/09_Shaft_Centerline.py — sidebar Historico SCL con
multiselect + extraccion usando compute_eccentricity_state +
tabla comparativa inline + diagnostico cuando CSVs no matchean
Sensor Map (similar a Bode).

(P3) _build_scl_report_notes acepta comparison_narrative
(estructura lista para narrativa modal en proxima iteracion
estilo Bently/API 670).

Diferido P4: overlay visual de trayectorias historicas sobre
SCL plot — requiere modificar render_scl_panel ~600 lineas.
Tabla comparativa actual muestra deltas numericos." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 17.3 (P1+P2+P3) pusheado a dev"
echo "================================================================"
echo ""
echo "Para verlo:"
echo "  1. SCL page → cargar CSVs."
echo "  2. Sidebar 'Historico SCL' → 📸 Guardar snapshot."
echo "  3. Cargar otra corrida y elegir snapshot anterior en el"
echo "     multiselect."
echo "  4. Tabla 'Comparativo SCL' arriba con Δ X/Y, Δ e/c, Δ"
echo "     attitude angle + diagnostico."
echo ""
echo "Pendiente proximo turno (17.3 P4):"
echo "  - Overlay visual de trails x/y historicos sobre el SCL plot"
echo "  - Narrativa modal completa SCL en PDF"
echo ""
echo "Estado del histórico multi-modulo:"
echo "  Tabular  ✅"
echo "  Polar    ✅ (con narrativa modal Bently/API 684)"
echo "  Bode     ✅ (con narrativa modal Bently/API 684)"
echo "  SCL      ✅ (snapshot + tabla; overlay+narrativa pendiente)"
echo "  Spectrum ⏳ (Ciclo 17.4)"
echo "================================================================"
