#!/bin/bash
# =============================================================
# Watermelon — HOTFIX Ciclo 15.1: panel debug + fix indentacion
# =============================================================
# El smoke test del matcher pasa pero la app real falla. Para
# diagnosticar EXACTAMENTE qué le llega al matcher por cada signal,
# agrego un panel de debug visible en Tabular List que muestra:
#
# Por cada signal cargado:
#   - CSV Point exacto
#   - CSV Variable
#   - CSV Unit
#   - Family inferido por el legacy code
#   - Sensor del map que matcheó (label) o "SIN MATCH"
#   - Tipo del sensor matched
#   - Unit native del sensor
#   - Alarm / Danger del sensor
#
# El expander está colapsado por default. Permite verificar de un
# vistazo si el matcher está funcionando con los datos reales del
# Bently sin tener que ir a la consola.
#
# Bug latente arreglado: machine_settings y point_settings quedaron
# DENTRO del 'if _override_active:' después del Ciclo 14c.3, lo que
# rompía cuando el usuario NO tenía override activo (sin caso edge,
# pero codigo defensivo).
#
# Ejecutar:
#   bash _hotfix_ciclo15_1_debug_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add pages/01__Tabular_List.py
git status --short | head

git commit -m "fix(tabular): hotfix Ciclo 15.1 — panel debug matching + fix indentacion

(1) Panel de debug visible (expander colapsado) en Tabular List que
muestra por cada signal cargado:
- CSV Point / Variable / Unit exactos que llegan al matcher
- Sensor del map que matcheo (label industrial) o 'SIN MATCH'
- Tipo / unit_native / alarm / danger del sensor matched

Permite diagnosticar de un vistazo si el matcher funciona con los
datos reales sin tener que ir a la consola.

(2) Fix de indentacion: machine_settings y point_settings quedaron
dentro del 'if _override_active:' tras el Ciclo 14c.3. Sacados al
nivel principal para que SIEMPRE se inicialicen, no solo cuando hay
override activo (codigo defensivo)."

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Hotfix listo"
echo "================================================================"
echo ""
echo "Pasos:"
echo "  1. Ctrl+C en la terminal de Streamlit y volver a arrancar"
echo "     'streamlit run 00_Home.py'"
echo "  2. Cargar los CSVs"
echo "  3. Ir a Tabular List"
echo "  4. Abrir el expander '🔍 Debug: matching de sensores...'"
echo "  5. Mandame screenshot del contenido del expander"
echo ""
echo "Eso me dice EXACTAMENTE qué Point/Variable/Unit le llega al"
echo "matcher por cada signal y por que falla (o no). Si los"
echo "velocimetros aparecen como 'SIN MATCH' o matched a un sensor"
echo "proximity en lugar de velocity, ahi vemos el origen."
