#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.1 Parte 2 → DEV: UI Polar history
# =============================================================
# Wire del polar_history (Parte 1) a la pagina Polar Plot.
# Ya el usuario PUEDE snapshotear corridas y ver comparativo
# diagnostico en la misma pagina.
#
# Cambios en pages/06_Polar_Plot.py:
#
# (A) Helper interno _wm_extract_polar_readings(items, sensors_map,
#     op_speed): para cada CSV polar, matched al sensor map via
#     resolve_sensor_for_point, y extrae amp/phase a la velocidad
#     operativa con nearest_row_for_speed. Devuelve lista lista
#     para save_polar_snapshot.
#
# (B) Nueva seccion sidebar "📚 Histórico Polar" debajo del
#     instance selector:
#       - Cantidad de snapshots para esta unidad
#       - Expander "📸 Guardar snapshot Polar actual" con etiqueta
#         + notas + boton guardar
#       - Selectbox "Comparar contra corrida anterior" con todos
#         los snapshots, marca con "(corrida actual)" los que sean
#         identicos a la sesion actual (skip_identical_to)
#       - Default cae sobre el primer snapshot DIFERENTE para que
#         el usuario vea comparativo real al toque
#       - Expander "🗂️ Gestionar snapshots" con borrar individual
#
# (C) Comparativo Polar inline en el cuerpo principal (arriba de
#     los paneles individuales) cuando hay snapshot anterior
#     elegido:
#       - Tabla por sensor: anterior amp + actual amp + Δ amp +
#         anterior fase + actual fase + Δ fase + diagnóstico
#       - Diagnostico humano con clasificacion del shift de fase:
#           shift_critical (>60°) = ⚠️ posible crack o falla severa
#           shift_major (≥30°)    = ⚠️ síntoma cambio de balance
#           shift_minor (10-30°)  = vigilar
#           stable (<10°)         = sin cambio
#       - Caption citando API 684 / ISO 21940-12
#
# Postpuesto a Parte 3 (proxima iteracion):
#   * Trail visual SOBRE el polar (overlay del snapshot anterior
#     en gris claro + flecha conectora al actual). Requiere
#     modificar build_polar_figure que es delicado.
#   * Seccion comparativa Polar en el PDF Reports.
#
# Ejecutar:
#   bash _publish_ciclo17_1_part2_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add pages/06_Polar_Plot.py
git status --short | head

git commit -m "feat(polar): UI snapshot + comparativo inline (Ciclo 17.1 P2)

Wire polar_history a la pagina Polar Plot. El usuario YA puede
snapshotear corridas y ver comparativo diagnostico inline.

(A) Helper _wm_extract_polar_readings: matched cada CSV a su
sensor del Sensor Map (resolve_sensor_for_point) y extrae amp/
phase a la velocidad operativa (nearest_row_for_speed).

(B) Sidebar 'Historico Polar' con snapshot button + label +
notas + selectbox comparar contra. Skip_identical_to marca
snapshots identicos a corrida actual. Default cae sobre el
primer DIFERENTE para mostrar comparativo real.

(C) Tabla comparativo inline arriba de los paneles cuando hay
snapshot anterior elegido. Por sensor: antes/ahora amp y fase,
Δamp+%, Δfase, diagnostico humano:
  shift_critical >60° = posible crack
  shift_major >=30°   = sintoma cambio balance (API 684)
  shift_minor 10-30°  = vigilar
  stable <10°         = sin cambio

Postpuesto Parte 3: trail visual sobre el polar mismo +
seccion comparativa Polar en PDF Reports." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 17.1 Parte 2 pusheado a dev"
echo "================================================================"
echo ""
echo "Para usarlo:"
echo "  1. Abrir Polar Plot, cargar CSVs, seleccionar Asset Instance."
echo "  2. Sidebar derecha → '📚 Histórico Polar' → expander '📸"
echo "     Guardar snapshot Polar actual' → etiqueta + guardar."
echo "  3. Cargar otra corrida (o cambiar archivos) y guardar otro"
echo "     snapshot."
echo "  4. En el dropdown 'Comparar contra corrida anterior' elegí"
echo "     el primero. Aparece arriba de los paneles la tabla"
echo "     'Comparativo Polar' con Δfase, Δamp y diagnóstico."
echo ""
echo "Próxima iteración (Parte 3):"
echo "  * Trail visual: overlay del snapshot anterior sobre el polar"
echo "    mismo (gris claro + flecha conectora al actual)."
echo "  * Sección comparativa Polar en el PDF Reports."
echo "================================================================"
