#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.1 Parte 1 → DEV: polar_history foundation
# =============================================================
# Modelo de datos + helpers para snapshot del modulo Polar.
# Esta es la BASE para la integracion UI en la pagina Polar
# Plot (proxima iteracion).
#
# core/polar_history.py NUEVO:
#
#   save_polar_snapshot(instance_id, operating_speed_rpm,
#       sensors_data=[{sensor_label, csv_file, amp_at_op,
#       phase_at_op, amp_unit, phase_unit, csv_timestamp}],
#       corrida_label, notes) -> snapshot_id
#
#   list_polar_snapshots, load_polar_snapshot, delete_polar_snapshot
#
#   get_polar_history_for_sensor(instance_id, sensor_label,
#       max_snapshots=8, current_reading=None) -> list de puntos
#       cronologicos. Acepta current_reading para anexar la corrida
#       actual al final si difiere del ultimo snapshot.
#
#   get_previous_polar_snapshot(instance_id,
#       skip_identical_to_sensors={label: {amp, phase}}) -> dict
#       Mismo patron que el Tabular: saltea snapshots cuyas
#       lecturas son esencialmente identicas a la corrida actual
#       (amp diff <1% y phase diff <1° por sensor).
#
#   phase_shift_classifier(delta_deg) -> categoria diagnostica:
#     stable          |Δ| < 10°
#     shift_minor     10° <= |Δ| < 30°
#     shift_major     30° <= |Δ| < 60°  (sintoma de cambio balance)
#     shift_critical  |Δ| >= 60°        (degradacion / crack)
#   Usa shortest-arc circular distance.
#
#   amplitude_change_classifier(delta_pct) -> amp_critical/high/up/
#     stable/down/down_strong segun los thresholds del Tabular.
#
#   shortest_arc_phase_diff(p1, p2) -> diff signed [-180, 180]
#     que indica direccion del shift.
#
# Storage local:
#   {INSTANCES_DIR}/{instance_id}/history/polar_{ISO8601}.json
#   Max 24 snapshots con auto-prune de los mas viejos.
#
# Smoke validado:
#   - 2 snapshots Polar (Abril 19 vs Abril 27)
#   - 3X_D: phase shift +35.5° clasificado shift_major
#   - 3X_D: amp +41.2% clasificado amp_high
#   - skip_identical_to_sensors detecta correctamente corrida
#     anterior salteando el snap actual identico
#
# Proxima iteracion: integracion UI en pages/06_Polar_Plot.py
# (boton snapshot + selector comparar contra + trail visual + PDF
# section).
#
# Ejecutar:
#   bash _publish_ciclo17_1_part1_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add core/polar_history.py
git status --short | head

git commit -m "feat(polar): polar_history foundation (Ciclo 17.1 P1)

Modelo de datos + helpers para snapshots del modulo Polar:
save/list/load/delete + get_polar_history_for_sensor +
get_previous_polar_snapshot con skip_identical_to_sensors +
clasificadores diagnosticos (phase_shift_classifier con
thresholds API 684 / shift_major >=30° = sintoma cambio balance,
amplitude_change_classifier).

Storage en {INSTANCES_DIR}/{id}/history/polar_{ISO}.json (max
24 con auto-prune). Reusa el mismo patron del Tabular history
para coherencia.

Esta es la base — la integracion UI en la pagina Polar Plot
viene en el proximo commit (selector comparar contra + trail
visual + seccion PDF)." || echo "Nothing to commit"

git push origin dev

echo ""
echo "✓ Base polar_history pusheada a dev. Sin cambios visibles aun."
echo "  Próxima iteración: UI snapshot + trail visual + PDF section."
