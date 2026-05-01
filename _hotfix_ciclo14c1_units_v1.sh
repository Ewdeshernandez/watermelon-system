#!/bin/bash
# =============================================================
# Watermelon — HOTFIX Ciclo 14c.1: unidades dropdown + match más amigable + bug fix
# =============================================================
# Tres mejoras pedidas + 1 bug encontrado:
#
# 1) UNIDAD como dropdown universal (NO TextColumn).
#    11 opciones agrupadas por familia:
#    - Desplazamiento: mil pp, µm pp, mm pp
#    - Velocidad: mm/s RMS, mm/s peak, in/s RMS, in/s peak
#    - Aceleración: g RMS, g peak, m/s² RMS, m/s² peak
#    Cuando "Generar mapa estandar" crea sensores, cada uno ya
#    viene con su unidad correcta. El usuario solo confirma.
#
# 2) MATCH CSV POINT renombrado a "Texto Point CSV", help mejorado.
#    Tres formatos válidos ahora:
#    - Substring simple: "VE5807" matchea "VE5807 (Y)"
#    - Lista por comas: "VE5807 (Y), VE5807-Y, 5807_Y" → OR
#    - Glob (back-compat): "*5807*y*"
#    Match case-insensitive en todos los formatos.
#
# 3) BUG ARREGLADO: heurística mal-clasificaba "mm/s" como
#    accelerometer.
#    Causa: "m/s" (substring del check de m/s²) era substring de
#    "mm/s" (velocidad). El orden if-elif chequeaba accelerometer
#    primero, así que falsos positivos.
#    Fix: chequear velocity ANTES (mm/s / in/s / ips), y para
#    accelerometer solo "m/s²" / "m/s2" / "g rms" / "g pk".
#    Validado: 'XYZ-UNKNOWN' con unit 'mm/s' ahora devuelve SIN
#    MATCH correctamente (antes devolvía falsamente el accelerometer).
#
# Ejecutar:
#   bash _hotfix_ciclo14c1_units_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add core/sensor_map.py pages/00_Machinery_Library.py
git status --short | head

git commit -m "fix(library): hotfix Ciclo 14c.1 — unidades dropdown + match amigable + bug heuristica

3 mejoras de UX/correctness pedidas tras testing inicial:

1) Columna 'Unidad' del data_editor ahora es SelectboxColumn con 11
   opciones agrupadas (3 desplazamiento + 4 velocidad + 4 aceleracion).
   Antes era TextColumn que requeria tipear la unidad exacta.

2) Columna 'Match CSV Point' renombrada a 'Texto Point CSV'. Acepta
   3 formatos: substring simple ('VE5807'), lista por comas
   ('VE5807 (Y), VE5807-Y, 5807_Y' = OR), o glob ('*5807*y*'). Help
   mejorado con ejemplos. Backward-compat con globs existentes.

3) Bug fix critico: heuristica de type_hint mal-clasificaba
   'mm/s' como accelerometer porque 'm/s' (substring para m/s²) es
   tambien substring de 'mm/s'. Causaba falsos positivos donde
   sensores velocity sin pattern matcheaban Points con unit g.
   Fix: chequear velocity ANTES (mm/s / in/s / ips) y para
   accelerometer solo m/s² / m/s2 / g rms / g pk.
   Smoke validado: 'XYZ-UNKNOWN' con unit 'mm/s' devuelve SIN MATCH."

git push origin dev

echo ""
echo "Refrescar app, ir a Machinery Library → Mapa de Sensores."
echo "  - Columna 'Unidad' ahora es dropdown con 11 opciones"
echo "  - Columna renombrada a 'Texto Point CSV' con help mejorado"
echo "  - Match heuristica ya no produce falsos positivos cross-familia"
