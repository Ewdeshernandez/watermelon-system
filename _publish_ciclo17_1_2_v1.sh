#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.1.2 → DEV: Trayectorias polares completas
# =============================================================
# Antes el snapshot guardaba solo el punto operativo (amp, phase
# a operating_rpm). Ahora guarda la TRAYECTORIA COMPLETA del run-
# up/coast-down de cada sensor — speed, amp, phase a lo largo de
# todo el rango de RPM, downsampleada a 80 puntos por sensor
# (~10KB por sensor por corrida en el JSON).
#
# Esto permite ver superpuestas las CORRIDAS COMPLETAS — incluido
# el paso por la velocidad crítica — entre snapshots, que es la
# vista canonica de Bently System 1 / API 684 / ISO 21940-12 para
# diagnóstico de balance del rotor.
#
# Ahora el polar muestra:
#
#   • TRAYECTORIA COMPLETA de cada snapshot anterior (line trace
#     a lo largo de speed, amp, phase) en color cronológico (azul
#     claro = más viejo, rojo = más reciente). Opacidad ~0.55.
#   • DIAMOND-OPEN marker en el peak (max amplitude) de cada
#     trayectoria histórica = velocidad crítica de esa corrida.
#     Tooltip con speed/amp/phase del peak.
#   • STAR-OPEN marker en el operating point de cada snapshot
#     con label de corrida + tooltip con Δamp y Δphase vs
#     actual (lo que ya tenía).
#   • La trayectoria actual (azul oscuro) y operating point
#     actual (estrella negra) sobre todo, sin tocar.
#
# Permite ver de un vistazo:
#
#   1. MODE MIGRATION: si el peak de la critica se desplaza en
#      RPM o cambia de fase entre corridas → cambio del modo
#      vibratorio del sistema rotor-soporte.
#   2. Q FACTOR DEGRADATION: si la amplitud del peak crece y/o
#      el bucle se vuelve más cerrado → menos amortiguamiento.
#   3. BALANCE SHIFT: si el operating point migra
#      independientemente del peak → masa del rotor cambió
#      (pérdida o crack).
#
# Backward-compat: snapshots viejos sin trajectory_* dibujan
# solo el operating point (linea conectora dotted como antes).
#
# Cambios:
#
# (1) core/polar_history.py — save_polar_snapshot acepta
#     trajectory_speed/amp/phase + critical_speed_rpm/amp/phase/
#     q_factor opcionales por sensor. Storage redondeado para
#     mantener JSON liviano.
#
# (2) pages/06_Polar_Plot.py:
#     - _wm_extract_polar_readings ahora downsamplea grouped_df
#       a 80 puntos uniformes y los incluye en cada entry.
#     - build_polar_figure dibuja trajectory completa + diamond
#       en el peak cuando trajectory_* está presente.
#     - render_polar_panel pasa trajectory_* desde el snapshot
#       cargado al figure builder.
#
# Alineado con: Bently Nevada Technical Training (Polar Vector
# History), API 684 Tutorial on Rotor Dynamics, ISO 21940-12
# Mechanical Vibration / Rotor Balancing.
#
# Ejecutar:
#   bash _publish_ciclo17_1_2_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add core/polar_history.py
git add pages/06_Polar_Plot.py
git status --short | head

git commit -m "feat(polar): trayectorias historicas completas superpuestas (Ciclo 17.1.2)

Snapshot guarda la trayectoria entera (speed, amp, phase
downsampleada a 80 puntos por sensor) ademas del operating
point. Render polar superpone los loops historicos completos
con gradiente cronologico (azul = mas viejo, rojo = mas
reciente), permitiendo ver paso por velocidad critica entre
corridas.

Cambios:
(1) core/polar_history.py: save_polar_snapshot acepta
trajectory_speed/amp/phase + critical_speed_* + q_factor
opcionales por sensor. JSON liviano (~10KB por sensor).
(2) pages/06_Polar_Plot.py: extraccion downsamplea
grouped_df a 80 puntos. build_polar_figure dibuja:
- Trajectory completa (line trace) por snapshot anterior con
  color cronologico y opacidad 0.55
- Diamond-open en el peak de amplitud = velocidad critica
  historica con tooltip (speed/amp/phase)
- Star-open en operating point como antes (legacy fallback
  para snapshots viejos sin trayectoria)

Vista canonica para diagnostico de balance segun Bently Nevada
Technical Training, API 684, ISO 21940-12:
- Mode migration (peak se desplaza)
- Q factor degradation (amplitud del peak crece)
- Balance shift (operating point migra)

Backward-compat: snapshots viejos sin trajectory dibujan solo
operating point como antes." || echo "Nothing to commit"

git push origin dev

echo ""
echo "✓ Ciclo 17.1.2 pusheado a dev. Refrescá."
echo ""
echo "OJO: los snapshots Polar viejos (los 3 que tenés) NO tienen"
echo "trayectoria — solo se va a ver el operating point como antes."
echo "Para que los nuevos tengan trayectoria completa, GUARDA un"
echo "nuevo snapshot ahora con la corrida cargada. El proximo"
echo "snapshot ya queda con trail completo, y los siguientes"
echo "tambien."
