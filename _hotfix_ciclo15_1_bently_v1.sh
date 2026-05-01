#!/bin/bash
# =============================================================
# Watermelon — HOTFIX Ciclo 15.1: Bently VT prevalece sobre unit
# =============================================================
# Bug residual: los velocimetros del Bently (Points "1VT6805 (C) TRF",
# "1VT6831 (C) CRF") seguian saliendo como Family=Proximity con
# unidad "mil pp" en Tabular List — incluso despues del hotfix
# anterior que agrego "VT" como hint.
#
# Causa raiz mas profunda:
#
# 1) Los velocity transducers Bently (VT5000 series) reportan
#    amplitude_unit en "mil pp" porque internamente integran a
#    displacement. Mi heuristica chequeaba la UNIT antes que el
#    POINT NAME → "mil" disparaba type_hint=proximity y nunca
#    llegaba a evaluar la pista del Point.
#
# 2) El generador estandar generaba patterns IGUALES para velocity
#    y acelerometer (ambos con prefix "acell" como sufijo). Resultado:
#    el sensor velocity tenia pattern "*acell*vel*" que NO matchea
#    Points reales como "1VT6805" (que NO contiene "acell").
#
# Fix doble en core/sensor_map.py:
#
# (a) HEURISTICA: detectar tipo por POINT NAME PRIMERO, unit como
#     respaldo solo si Point name no fue conclusive. Tokens VT/VEL
#     fuerzan velocity AUN si la unit dice "mil pp".
#
# (b) GENERADOR para modo accel_plus_velocity:
#     - Sensor accel: pattern usa "*acell*" / "*acc*"
#     - Sensor velocity: pattern usa "*vt*" / "*vel*" (Bently typical)
#     Con prefix configurable distinto por sensor.
#
# Smoke validado contra el sensor map REAL del usuario (CRF/TRF +
# GEN DE/NDE + keyphasor) con Points Bently reales:
#
#   1VT6805 (C) TRF → 2_RAD_V (velocity, TRF Vel) ✓
#   1VT6831 (C) CRF → 1_RAD_V (velocity, CRF Vel) ✓
#   TRF ACELL       → 2_RAD_A (accel, TRF Accel) ✓
#   CRF ACELL       → 1_RAD_A (accel, CRF Accel) ✓
#   VE5807-VE5810   → 3X_D / 3Y_D / 4X_D / 4Y_D (proximity) ✓
#   KPHGEN          → 1_RAD_K (keyphasor) ✓
#
# Ejecutar:
#   bash _hotfix_ciclo15_1_bently_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add core/sensor_map.py
git status --short | head

git commit -m "fix(sensor-map): hotfix Ciclo 15.1 — Bently VT prevalece sobre unit + patterns velocity correctos

Bug critico: velocimetros Bently (1VT6805 / 1VT6831) seguian saliendo
como Proximity con mil pp en Tabular List, incluso despues del hotfix
previo de heuristica.

Causa raiz dual:

1) Los VT Bently reportan amplitude_unit en 'mil pp' (integran a
   displacement internamente). La heuristica chequeaba unit ANTES que
   point name → 'mil' disparaba type_hint=proximity antes de ver 'VT'.

2) El generador estandar para modo accel_plus_velocity ponia el mismo
   prefix ('acell') en patterns de velocity Y accel. Resultado: pattern
   del velocity era '*acell*vel*' que NO matchea Points Bently reales
   como '1VT6805 (C) TRF'.

Fix doble:
* core/sensor_map.resolve_sensor_for_point: detectar tipo por POINT
  NAME primero (1VT/2VT/VT/VEL/ACELL), unit como respaldo solo si
  Point name no concluyente. Tokens VT/VEL fuerzan velocity aunque
  la unit diga mil.
* core/sensor_map.generate_standard_sensor_map (modo accel_plus_velocity):
  * accel pattern: '*{prefix}*acell*, *{prefix}*acc*'
  * velocity pattern: '*vt*{prefix}*, *{prefix}*vel*, *{prefix}*vt*'
  Cubre nomenclatura Bently real (VT prefix).

Smoke validado contra sensor map del usuario con CRF/TRF + GEN DE/NDE
+ keyphasor → todos los Points reales (1VT6805/1VT6831/TRF ACELL/
CRF ACELL/VE5807-5810/KPHGEN) matchean al sensor correcto."

git push origin dev

echo ""
echo "Refrescar app y validar Tabular List:"
echo "  Filas 1VT6805 / 1VT6831 ahora salen como Family=Velocity con"
echo "  in/s peak (o lo que tengas en sensor map), Alarm 1.500,"
echo "  Danger 2.000."
echo "  CRF ACELL / TRF ACELL siguen Acceleration con g peak."
echo "  Proximities VE5807-VE5810 siguen Proximity con mil pp."
