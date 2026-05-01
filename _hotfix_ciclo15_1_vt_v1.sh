#!/bin/bash
# =============================================================
# Watermelon — HOTFIX Ciclo 15.1: matcher reconoce VT y tokens TRF/CRF
# =============================================================
# Bug en Tabular List: los velocímetros del Bently con Points como
# "1VT6805 (C) TRF" salían clasificados como Family=Proximity con
# unidad "mil pp" porque el matcher heurístico no detectaba "VT"
# (Velocity Transducer en lenguaje Bently/Connecticut) como hint
# de velocity, y el tie-break por plane_label no encontraba "TRF"
# embebido en el Point.
#
# Fix doble en core/sensor_map.py:
#
# 1) HEURÍSTICA por SUBSTRING DEL POINT (no solo unit):
#    * "vt" / "vel" / "velo" en point → type_hint = velocity
#    * "acell" / "accel" / "ace" → type_hint = accelerometer
#    * "ve5..." / "(x)" / "(y)" / "disp" / "dsp" → type_hint = proximity
#    Esto cubre nombres reales del Bently System 1 además de los
#    nombres genéricos.
#
# 2) TIE-BREAK MÁS PERMISIVO por tokens cortos del plane_label:
#    Antes solo matcheaba si el plane_label completo era substring
#    del Point. Ahora splitea el plane_label por whitespace/paréntesis
#    y busca tokens distintivos (TRF, CRF, BRG, etc.) ignorando
#    palabras comunes (DE, NDE, LM, TM, brush, driver, driven).
#
# Smoke validado contra Points reales de TES1:
#   1VT6805 (C) TRF → 2_RAD_V (velocity, plane TRF)
#   1VT6831 (C) CRF → 1_RAD_V (velocity, plane CRF)
#   TRF ACELL       → 2_RAD_A (accel, plane TRF)
#   CRF ACELL       → 1_RAD_A (accel, plane CRF)
#   VE5807 (Y)      → 3Y_D    (proximity)
#   VE5808 (X)      → 3X_D    (proximity)
#   KPHGEN          → 0_AX_K  (keyphasor)
#
# Ejecutar:
#   bash _hotfix_ciclo15_1_vt_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add core/sensor_map.py
git status --short | head

git commit -m "fix(sensor-map): hotfix Ciclo 15.1 — matcher reconoce VT y tokens cortos del plane_label

Bug: velocimetros del Bently con Points 1VT6805 / 1VT6831 caian como
Family=Proximity con mil pp. La heuristica buscaba 'vel' / 'velo' pero
no 'vt' (Velocity Transducer Bently). Y el tie-break solo matcheaba el
plane_label completo, no tokens cortos como 'TRF' o 'CRF'.

Fix doble:
* Heuristica por substring del Point (no solo unit):
  - 'vt' / 'vel' / 'velo' → velocity
  - 'acell' / 'accel' / 'ace' → accelerometer
  - 've5...' / '(x)' / '(y)' / 'disp' / 'dsp' → proximity
* Tie-break con tokens cortos del plane_label (split por whitespace
  y parentesis, ignora DE/NDE/LM/brush/etc).

Smoke validado: TES1 real con Points 1VT6805 (TRF), 1VT6831 (CRF),
TRF ACELL, CRF ACELL, VE5807-VE5810, KPHGEN → todos matchean al
sensor correcto."

git push origin dev

echo ""
echo "Refrescar app y validar en Tabular List:"
echo "  - Filas de 1VT6805 / 1VT6831 ahora salen como Family=Velocity"
echo "    con unidad 'mm/s RMS' (o lo que tengas en el sensor map)"
echo "  - Alarm/Danger del sensor velocity (no del proximity global)"
echo "  - El criterion correspondiente"
