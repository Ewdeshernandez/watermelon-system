#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.2 → DEV: Bode history completo
# =============================================================
# Aplica el patrón validado del Ciclo 17.1 (Polar history) al
# módulo Bode. El usuario puede ahora snapshotear corridas Bode
# (críticas, Q factor, op point + trayectoria amp/fase vs RPM)
# y superponer múltiples corridas históricas sobre el Bode plot
# con gradiente cronológico.
#
# Permite ver al toque:
#
#   - MIGRACIÓN DEL MODO: peak de la crítica desplazándose en
#     RPM entre corridas → mode shift (pérdida de stiffness en
#     soportes, cambio de geometría)
#   - DEGRADACIÓN DEL Q FACTOR: amplitud del peak crece sin que
#     migre la frecuencia → pérdida de damping hidrodinámico
#   - DERIVA DE FASE A TRAVÉS DEL MODO: cambia el delta phase
#     a través del peak → cambio del modo de respuesta
#     vibratoria del rotor
#
# Cambios:
#
# (P1) core/bode_history.py NUEVO — paralelo a polar_history,
#     archivos bode_*.json. Helpers: save/list/load/delete +
#     get_bode_history_for_sensor + get_previous_bode_snapshot
#     con skip_identical_to_sensors. Reusa los clasificadores
#     de polar_history (phase_shift_classifier,
#     amplitude_change_classifier, shortest_arc_phase_diff) que
#     son módulo-agnósticos.
#
# (P2) pages/07_Bode_Plot.py — sidebar "Histórico Bode" con
#     multiselect + extracción downsampleada de trayectoria
#     (80 puntos por sensor) + tabla comparativa inline +
#     overlay de curvas amp/phase vs RPM con gradiente
#     cronológico (azul claro = más viejo, rojo = más reciente)
#     + diamond-open marker en cada peak histórico = velocidad
#     crítica de esa corrida.
#
# (P3) Narrativa modal completa estilo Bently/API 684 inyectada
#     en text_diag["comparison_narrative"] del Bode. 5 bloques:
#     (1) Encabezado factual con vector change
#     (2) Caracterización del modo (translacional/cónico/
#         flexural por phase delta a la crítica) +
#         comparativo de critical speed entre snapshots
#     (3) Diagnóstico diferencial del shift de fase
#     (4) Análisis de sensitividad vectorial / damping
#     (5) Distinción modal rotor vs estructural
#
#     _build_bode_report_notes incluye comparison_narrative
#     entre el detalle del diagnóstico y las acciones
#     recomendadas — flujo natural del reporte ingenieril.
#
# Vocabulario alineado con Bently Nevada Technical Training,
# API 684 Tutorial on Rotor Dynamics, ISO 21940-12.
#
# Ejecutar:
#   bash _publish_ciclo17_2_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add core/bode_history.py
git add pages/07_Bode_Plot.py
git status --short | head

git commit -m "feat(bode): Bode history completo (Ciclo 17.2)

Aplica el patron validado del Polar history (17.1) al modulo Bode.
Snapshot de criticas + Q + op point + trayectoria amp/fase vs RPM.
Multi-snapshot overlay con gradiente cronologico sobre el Bode
permite ver migracion del modo, degradacion del Q factor y deriva
de fase a traves del modo entre corridas.

(P1) core/bode_history.py NUEVO — paralelo a polar_history,
archivos bode_*.json. save/list/load/delete + get_bode_history_for_sensor
+ get_previous_bode_snapshot con skip_identical. Reusa
clasificadores agnosticos (phase_shift_classifier,
amplitude_change_classifier, shortest_arc_phase_diff).

(P2) pages/07_Bode_Plot.py:
- Sidebar 'Historico Bode' con multiselect, snapshot button,
  extraccion downsampleada (80 puntos por sensor) y comparativo
  inline en tabla (sensor x corrida).
- build_bode_figure acepta prev_snapshots, dibuja overlays amp y
  phase vs RPM con gradiente azul-ambar-rojo + diamond-open en
  cada peak historico (= critical speed por corrida).
- render_bode_panel busca snapshots elegidos, resuelve sensor
  matched al panel y arma lista para el overlay.

(P3) Narrativa modal completa en PDF Reports — 5 bloques estilo
Bently/API 684 inyectada en comparison_narrative:
(1) encabezado factual con vector change
(2) caracterizacion del modo + critical speed migration
(3) diagnostico diferencial del shift
(4) analisis de sensitividad / damping
(5) distincion modal rotor vs estructural

_build_bode_report_notes incluye comparison_narrative entre
detalle y acciones." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 17.2 completo pusheado a dev"
echo "================================================================"
echo ""
echo "Para verlo end-to-end:"
echo "  1. Abrir Bode Plot, cargar CSVs."
echo "  2. Sidebar 'Historico Bode' → Guardar snapshot."
echo "  3. Cargar otra corrida + Comparar contra anterior."
echo "  4. El Bode plot ahora muestra:"
echo "     - Línea actual (azul oscuro) en amp vs RPM y phase vs RPM"
echo "     - Curvas históricas con gradiente cronológico debajo"
echo "     - Diamond-open en peak de cada corrida histórica"
echo "  5. Tabla 'Comparativo Bode' arriba con Δamp + Δfase + estado"
echo "  6. Send to Report → narrativa modal completa en el PDF"
echo ""
echo "Ya tenemos historico para Tabular + Polar + Bode. Próximas:"
echo "  - 17.3 SCL history (X/Y migration + eccentricity ratio)"
echo "  - 17.4 Spectrum history (peaks por orden)"
echo "  - O publicar v2.6 a main con todo lo acumulado."
echo "================================================================"
